from math import ceil, log10, prod

from zigzag.datatypes import LayerDim, LayerOperand, UnrollFactor

from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode

MODULATION_NUMBER = 10**3  # Must be higher than any node's sub id


def nearest_power_of_10(x: int):
    return 10 ** ceil(log10(x))


def convert_id(i: int, j: int) -> int:
    assert 0 <= j < MODULATION_NUMBER
    return MODULATION_NUMBER * i + j


def invert_id(k: int) -> tuple[int, int]:
    i, j = divmod(k, MODULATION_NUMBER)
    return i, j


def convert_ids(nodes: list[ComputationNode]):
    ids: dict[ComputationNode, int] = {}
    for node in nodes:
        i, j = node.id, node.sub_id
        new_id = convert_id(i, j)
        ids[node] = new_id
    return ids


def invert_ids_list(ids_list: list[tuple[int, int, int]], nb_nodes: int) -> list[tuple[int, int, tuple[int, int]]]:
    new_l: list[tuple[int, int, tuple[int, int]]] = []
    for slot, core, k in ids_list:
        new_l.append((slot, core, invert_id(k)))
    return new_l


def get_loop_size(loops: list[tuple[LayerDim, UnrollFactor]], dims: list[LayerDim]) -> int:
    return int(prod([tl[1] for tl in loops if tl[0] in dims]))


def get_latencies(
    nodes: list[ComputationNode],
    core_ids: list[int],
    accelerator: Accelerator,
    cost_lut: CostModelEvaluationLUT,
    impossible_lat: float = 1e11,
    ids: dict[ComputationNode, int] = {},
) -> tuple[dict[tuple[int, str, int], int], dict]:
    if not ids:
        ids = {node: node.id for node in nodes}
    core_names = [f"Core {id}" for id in core_ids]
    latencies = {(ids[node], core_name): impossible_lat for node in nodes for core_name in core_names}
    possible_allocations: dict[int, list[str]] = {}
    inter_core_tiling_sizes = {}

    for node in nodes:
        node_id = ids[node]
        possible_allocations[node_id] = []
        for core_id, core_name in zip(core_ids, core_names):
            core = accelerator.get_core(core_id)
            try:
                equal_node = cost_lut.get_equal_node(node)
                assert equal_node, f"No equal node for {node} found in CostModelEvaluationLUT"
                cme = cost_lut.get_cme(equal_node, core)
                output_operand = LayerOperand("O")
                temporal_loops = [
                    i for tm_level in cme.temporal_mapping.mapping_dic_stationary[output_operand] for i in tm_level
                ]
                inter_core_tiling_dims = [layer_dim for layer_dim, _ in node.inter_core_tiling]
                inter_core_tiling_size = get_loop_size(temporal_loops, inter_core_tiling_dims)
                inter_core_tiling_sizes[(node_id, core_name)] = inter_core_tiling_size
                lat = cme.latency_total1
                possible_allocations[node_id].append(core_name)
            except ValueError:
                lat = impossible_lat
            latencies[(node_id, core_name)] = lat

    latencies_with_split = {}
    possible_allocation_splits = {}
    p_max = len(core_names)  # maximum parallalization factor

    for node_id in ids.values():
        possible_allocation_splits[node_id] = {}
        for core_name in core_names:
            possible_allocation_splits[node_id][core_name] = {}
            if core_name in possible_allocations[node_id]:
                p_t = int(inter_core_tiling_sizes[node_id, core_name])
                for p in range(1, p_max + 1):
                    if divmod(p_t, p)[1] == 0 and p <= len(possible_allocations[node_id]):
                        lat = int(latencies[(node_id, core_name)] / min(p_t, p))
                        possible_allocation_splits[node_id][core_name][p] = 1
                    else:
                        lat = impossible_lat
                        possible_allocation_splits[node_id][core_name][p] = 0
                    latencies_with_split[(node_id, core_name, p)] = lat
            else:
                for p in range(1, p_max + 1):
                    latencies_with_split[(node_id, core_name, p)] = impossible_lat
                    possible_allocation_splits[node_id][core_name][p] = 0

    return latencies_with_split, possible_allocation_splits


def get_energies(
    nodes: list[ComputationNode],
    core_ids: list[int],
    accelerator: Accelerator,
    cost_lut: CostModelEvaluationLUT,
    impossible_energy: float = 1e11,
    ids: dict[ComputationNode, int] = {},
) -> dict[tuple[int, str], float]:
    if not ids:
        ids = {node.id: node.id for node in nodes}
    core_names = [f"Core {id}" for id in core_ids]
    energies = {(ids[node], core_name): impossible_energy for node in nodes for core_name in core_names}

    for node in nodes:
        for core_id, core_name in zip(core_ids, core_names):
            core = accelerator.get_core(core_id)
            try:
                cme = cost_lut.get_cme(node, core)
                en = getattr(cme, "energy_total")
            except ValueError:
                en = impossible_energy
            energies[(ids[node], core_name)] = en

    return energies
