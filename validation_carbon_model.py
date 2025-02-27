from typing import Any

class CarbonModel: 
   def __init__(
        self,
        CI_op: int, 
        CI_em: int, 
        lifetime: int, 
        frequency: float,
        technology_node: dict[str, Any] | list[dict[str, Any]],
        area_list: dict[str, Any] | list[dict[str, Any]],
        energy_use:int
    ):
        self.CI_op = CI_op
        self.CI_em = CI_em
        self.lifetime = lifetime
        self.frequency = frequency
        self.technology_node = technology_node
        self.area_list = area_list
        self.energy_use = energy_use