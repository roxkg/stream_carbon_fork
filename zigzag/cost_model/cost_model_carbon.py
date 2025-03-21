import logging

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.hardware.architecture.accelerator import Accelerator
from zigzag.hardware.architecture.carbonparam import CarbonParam
from zigzag.mapping.spatial_mapping_internal import SpatialMappingInternal
from zigzag.mapping.temporal_mapping import TemporalMapping
from zigzag.utils import json_repr_handler
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class CostModelEvaluationForCarbon(CostModelEvaluation):
    """! Class that stores inputs and runs them through the zigzag cost model.

    Initialize the cost model evaluation with the following inputs:
    - accelerator: the accelerator that includes the core on which to run the layer
    - layer: the layer to run
    - spatial_mapping: the spatial mapping
    - temporal_mapping: the temporal mapping

    From these parameters, the following attributes are computed:
    * core: The core on which the layer is ran. This should be specified in the LayerNode attributes.
    * mapping: The combined spatial and temporal mapping object where access patterns are computed.

    The following cost model attributes are also initialized:
    - mem_energy_breakdown: The energy breakdown for all operands
    - energy: The total energy

    After initialization, the cost model evaluation is run.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        spatial_mapping_int: SpatialMappingInternal,
        temporal_mapping: TemporalMapping,
        carbonparam: CarbonParam,
        access_same_data_considered_as_no_access: bool = True
    ):
        self.carbomparam = carbonparam
        self.task_num: int = 1
        self.carbon_per_task: float = 0.0
        self.carbon_total: float = 0.0
        super().__init__(
            accelerator=accelerator,
            layer=layer,
            spatial_mapping=spatial_mapping,
            spatial_mapping_int=spatial_mapping_int,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access,
        )

    def run(self) -> None:
        """! Run the cost model evaluation."""
        super().calc_memory_utilization()
        super().calc_memory_word_access()
        self.calc_energy()
        self.calc_latency()
        self.calc_carbon()

    def calc_carbon(self) ->None: 
        """
        this function returns operational carbon based on calculated latency and energy cost
        ope carbon = lifetime * on ratio/latency * energy per task
        """
        self.task_num = self.carbomparam.lifetime * self.carbomparam.on_ratio /self.latency_total2
        self.carbon_per_task = self.energy_total * self.carbomparam.CI_op
        self.carbon_total = self.carbon_per_task * self.task_num

    def __jsonrepr__(self):

        return json_repr_handler(
            {
                "outputs": {
                    "memory": {
                        "utilization": (self.mem_utili_shared if hasattr(self, "mem_utili_shared") else None),
                        "word_accesses": self.memory_word_access,
                    },
                    "energy": {
                        "energy_total": self.energy_total,
                        "operational_energy": self.mac_energy,
                        "memory_energy": self.mem_energy,
                        "memory_energy_breakdown_per_level": self.mem_energy_breakdown,
                        "memory_energy_breakdown_per_level_per_operand": self.mem_energy_breakdown_further,
                    },
                    "latency": {
                        "data_onloading": self.latency_total1 - self.latency_total0,
                        "computation": self.latency_total0,
                        "data_offloading": self.latency_total2 - self.latency_total1,
                    },
                    "carbon": {
                        "task_num": self.task_num,
                        "carbon_per_task": self.carbon_per_task, 
                        "carbon_total": self.carbon_total                        
                    },
                    "spatial": {
                        "mac_utilization": {
                            "ideal": self.mac_spatial_utilization,
                            "stalls": self.mac_utilization0,
                            "stalls_onloading": self.mac_utilization1,
                            "stalls_onloading_offloading": self.mac_utilization2,
                        }
                    },
                },
                "inputs": {
                    "accelerator": self.accelerator,
                    "layer": self.layer,
                    "spatial_mapping": (self.spatial_mapping_int if hasattr(self, "spatial_mapping_int") else None),
                    "temporal_mapping": (self.temporal_mapping if hasattr(self, "temporal_mapping") else None),
                },
            }
        )

    def __simplejsonrepr__(self) -> dict[str, float]:
        """! Simple JSON representation used for saving this object to a simple json file."""
        return {
            "energy": self.energy_total,
            "latency": self.latency_total2,
            "carbon": self.carbon_total
        }
