from typing import Any
from zigzag.utils import open_yaml

from stream.hardware.architecture.carbonparam import CarbonParam

class CarbonFactory: 
    """
    Convert carbon data into CarbonParam instance
    """

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def create(self) -> CarbonParam: 
        """
        Create an CarbonParam instance from the user-provided data.
        """
        # CI_path = "stream/inputs/examples/carbon/carbon_intensity.yaml",
        carbon_intensity_data = open_yaml("stream/inputs/examples/carbon/carbon_intensity.yaml")
        src_op = self.data["CI_em"]
        CI_em = carbon_intensity_data.get(src_op, 0)
        src_op = self.data["CI_op"]
        CI_op = carbon_intensity_data.get(src_op, 0)
        return CarbonParam(CI_em= CI_em, CI_op= CI_op, lifetime= self.data["lifetime"]*self.data["on_ratio"], 
                           frequency=self.data["frequency"], technology_node=self.data["technology_node"])
