import logging
from typing import Any

from zigzag.utils import open_yaml

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.carbon_validator import CarbonValidator
from stream.parser.carbon_factory import CarbonFactory
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)

class CarbonParamParserStage(Stage): 
    """Parse to parse carbon parameter from a user-defined yaml file."""
    
    def __init__(self, 
        list_of_callables: list[StageCallable], 
        *, 
        carbon_path: str,
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        assert carbon_path.split(".")[-1] == "yaml", "Expected a yaml file as carbon parameter input"
        # build CarbonParam based on input yaml
        carbon_data = open_yaml(carbon_path)
        validator = CarbonValidator(carbon_data, carbon_path)
        self.carbonparam_data = validator.normalized_data  # store data after validation
        validate_success = validator.validate()
        if not validate_success:
            raise ValueError("Failed to validate user provided accelerator.")
        factory = CarbonFactory(self.carbonparam_data)
        self.carbonparam = factory.create()  # create CarbonParam instance

    def run(self): 
        self.kwargs["carbon_param"] = self.carbonparam
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=self.accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info