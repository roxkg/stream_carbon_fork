from zigzag.datatypes import LayerOperand
from zigzag.utils import pickle_deepcopy

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.carbonparam import CarbonParam
from stream.utils import CostModelEvaluationLUT, get_required_offchip_bandwidth, get_too_large_operands
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload


class FitnessEvaluator:
    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        carbon_param: CarbonParam,
        embodied_carbon: float,
        cost_lut: CostModelEvaluationLUT,
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.carbon_param = carbon_param
        self.embodied_carbon= embodied_carbon
        self.cost_lut = cost_lut
        # self.num_cores = len(inputs.accelerator.cores)

    def get_fitness(self):
        raise NotImplementedError


class StandardFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally."""

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        carbon_param: CarbonParam,
        embodied_carbon: float, 
        cost_lut: CostModelEvaluationLUT,
        layer_groups_flexible,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
    ) -> None:
        super().__init__(workload, accelerator, carbon_param, embodied_carbon, cost_lut)

        # self.weights = (-1.0, -1.0, 0)
        self.weights = (-1.0, )
        self.metrics = ["tCDP"]
        #self.metrics = ["carbon", "energy", "latency", ]
        #self.weights = (-1.0,)
        #self.metrics = ["carbon"]
        # self.weights = (-1.0, -1.0)
        # self.metrics = ["CD", "ED"]
        self.layer_groups_flexible = layer_groups_flexible
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            pickle_deepcopy(self.carbon_param),
            pickle_deepcopy(self.embodied_carbon),
            self.operands_to_prefetch,
            self.scheduling_order,
        )
        scme.run()
        energy = scme.energy
        latency = scme.latency
        carbon = scme.carbon
        CD = scme.CD
        ED = scme.ED
        tCDP = scme.tCDP
        # if not return_scme:
        #     return energy, latency, carbon
        # return energy, latency, carbon, scme
        
        if not return_scme:
            return (tCDP,)
        return (tCDP,) , scme
        
        
        # if not return_scme:
        #     return CD,ED
        # return CD,ED , scme

    def set_node_core_allocations(self, core_allocations: list[int]):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core
        allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            (layer_id, group_id) = self.layer_groups_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (
                node
                for node in self.workload.node_list
                if isinstance(node, ComputationNode) and node.id == layer_id and node.group == group_id
            )
            for node in nodes:
                equal_unique_node = self.cost_lut.get_equal_node(node)
                assert equal_unique_node is not None, "Node not found in CostModelEvaluationLUT"
                cme = self.cost_lut.get_cme(equal_unique_node, core)
                onchip_energy = cme.energy_total  # Initialize on-chip energy as total energy
                latency = cme.latency_total1
                too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core_allocation)
                # If there is a too_large_operand, we separate the off-chip energy.
                offchip_energy = 0
                for too_large_operand in too_large_operands:
                    layer_operand = next(
                        (k for (k, v) in cme.layer.memory_operand_links.data.items() if v == too_large_operand)
                    )
                    layer_operand_offchip_energy = cme.mem_energy_breakdown[layer_operand][-1]
                    offchip_energy += layer_operand_offchip_energy
                    onchip_energy -= layer_operand_offchip_energy
                # If there was offchip memory added for too_large_operands, get the offchip bandwidth
                required_offchip_bandwidth = get_required_offchip_bandwidth(cme, too_large_operands)
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(int(latency))
                node.set_chosen_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(required_offchip_bandwidth)
