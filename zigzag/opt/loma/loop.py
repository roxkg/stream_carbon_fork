from zigzag.datatypes import LayerDim, UnrollFactor


class Loop:
    """! Class that stores a single for-loop's information."""

    def __init__(self, layer_dim: LayerDim, size: UnrollFactor, loop_type: str = "temporal"):
        """! Initialize the loop with the given layer_dim string and size"""
        self.layer_dim = layer_dim
        self.size = size
        self.loop_type = loop_type

    def __str__(self):
        return f"{self.loop_type.capitalize()}Loop({self.layer_dim},{self.size})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Loop):
            return False
        return self.layer_dim == other.layer_dim and self.size == other.size and self.loop_type == other.loop_type
