import numpy as np
from zigzag.utils import open_yaml

sram_scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
scalings = sram_scaling["energy"]
nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
nodes = nodes["technology_node"]
nodes = nodes[:-1]

inputs = [0.0540043, 0.0560527]
tech = 16
output = [input/1.997275204*np.interp(tech, nodes, scalings) for input in inputs]
print(output)