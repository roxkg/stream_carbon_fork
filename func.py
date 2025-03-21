import json
import yaml

# JSON 数据
json_data = {
    "area":       [ 1,  1,           1.902779873, 3.757501843,  4.336197791],
    "delay":      [ 1,  1.25268817,  1.562502831, 1.959590643,  2.464222503],
    "energy":     [ 1,  1,           1.26597582,  1.997275204,  2.468013468],
    "edp":        [ 1,  1.075268817, 1.433721201, 2.335994391,  3.120118164],
    "power":      [ 1,  1,           1.231707317, 1.844748858,  2.154666667],
    "throughput": [ 1,  0.925925926, 0.883392226, 0.851788756,  0.788643533]
    }

# 写入 YAML 文件
with open("sram_scaling.yaml", "w") as yaml_file:
    yaml.dump(json_data, yaml_file, default_flow_style=False)

print("JSON 数据已成功转换为 YAML")
