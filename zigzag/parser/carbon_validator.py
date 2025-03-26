import logging
import os
from typing import Any

from cerberus import Validator

logger = logging.getLogger(_name_)

class CarbonValidator: 
    INPUT_DIR_LOCATION = "zigzag/inputs/"
    SCHEMA: dict [str, Any] = {
        "CI_op": {"type": "string", "allowed": ["Coal", "Gas", "Biomass", "Solar", "Geothermal", "Hydropower", "Nuclear", "Wind"]},
        "CI_em": {"type": "string", "allowed": ["Coal", "Gas", "Biomass", "Solar", "Geothermal", "Hydropower", "Nuclear", "Wind"]},
        "lifetime": {"type": "integer", "default": 0},
        "frequency":{"type": "float", "default": 1},
        "technology_node":{"type": "integer", "default": 28, "min": 7, "max": 28},
        "on_ratio":{"type": "float", "min": 0, "max": 1}, 
        "package_type":{"type": "string", "allowed": ["RDL", "active", "3D", "passive" ,"EMIB"]}
    }
    
    def _init_(self, data: Any, carbon_path: str):
        self.validator = Validator()
        self.validator.schema = CarbonValidator.SCHEMA
        self.data: dict[str, Any] = self.validator.normalized(data)
        self.is_valid = True
        self.carbon_dirname = os.path.dirname(carbon_path)

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined carbon parameter is invalid. %s", extra_msg)

    def validate(self) -> bool:
        """! Validate the user-provided carbon data. Log a critical warning when invalid data is encountered and
        return true iff valid.
        """
        # Validate according to schema
        validate_success = self.validator.validate(self.data)  # type: ignore
        errors = self.validator.errors  # type: ignore
        if not validate_success:
            self.invalidate(f"The following restrictions apply: {errors}")

        return self.is_valid

    @property
    def normalized_data(self) -> dict[str, Any]:
        """Returns the user-provided data after normalization by the validator. (Normalization happens during
        initialization)"""
        return self.data