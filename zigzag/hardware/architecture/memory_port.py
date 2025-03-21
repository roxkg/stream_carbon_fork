import re
from enum import StrEnum
from typing import Any, TypeAlias

from zigzag.datatypes import Constants, MemoryOperand
from zigzag.parser.accelerator_validator import AcceleratorValidator


class MemoryPortType(StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class DataDirection(StrEnum):
    RD_OUT_TO_LOW = "rd_out_to_low"
    WR_IN_BY_LOW = "wr_in_by_low"
    RD_OUT_TO_HIGH = "rd_out_to_high"
    WR_IN_BY_HIGH = "wr_in_by_high"


OperandDirection: TypeAlias = tuple[MemoryOperand, int, DataDirection]
PortAllocUserFormat: TypeAlias = tuple[dict[str, str], ...]


class MemoryPort:
    """Single port of a MemoryInstance"""

    port_id_counter = 0

    def __init__(
        self,
        port_name: str,
        type: MemoryPortType,
        bandwidth_min: int,
        bandwidth_max: int,
        port_id: int | None = None,
    ):
        """
        Collect all the physical memory port related information here.
        @param port_name:
        @param bandwidth_min: bit/cc
        @param bandwidth_max: bit/cc
        @param type: read_only (read), write_only (write), read_write (read_write)
        @param port_id: port index per memory
        """
        self.name = port_name
        self.bw_min = bandwidth_min
        self.bw_max = bandwidth_max
        self.type = type
        self.served_op_lv_dir: list[OperandDirection] = []

        #  to give each port a unique id number
        if port_id is None:
            self.port_id = MemoryPort.port_id_counter
            MemoryPort.port_id_counter += 1
        else:
            self.port_id = port_id
            MemoryPort.port_id_counter = port_id + 1

    def add_port_function(self, operand_level_direction: OperandDirection):
        self.served_op_lv_dir.append(operand_level_direction)

    @property
    def port_is_shared_by_two_input_operands(self):
        served_operands = set(s[0] for s in self.served_op_lv_dir if s[0] in [Constants.MEM_OP_1, Constants.MEM_OP_2])
        return len(served_operands) > 1

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, MemoryPort)
            and self.bw_max == other.bw_max
            and self.bw_min == other.bw_min
            and self.type == other.type
        )

    def __hash__(self):
        return self.port_id


class PortAllocation:
    """Port allocation for a single memory instance. Stores which ports are available for which memory operands and
    their corresponding direction.
    """

    def __init__(self, data: dict[MemoryOperand, dict[DataDirection, str]]):
        assert all(
            [
                all([isinstance(v, str) and re.match(AcceleratorValidator.PORT_REGEX, v) for v in d.values()])
                for d in data.values()
            ]
        )
        self.data = data

    def get_alloc_for_mem_op(self, mem_op: MemoryOperand) -> dict[DataDirection, str]:
        return self.data[mem_op]
