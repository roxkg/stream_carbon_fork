import json
import yaml

# JSON 数据
json_data = {
    "beolVfeol" : [ 0.5090, 0.4887, 0.4916, 0.4855, 0.5675, 0.5675]
}


# 写入 YAML 文件
with open("beolVfeol_scaling.yaml", "w") as yaml_file:
    yaml.dump(json_data, yaml_file, default_flow_style=False)

print("JSON 数据已成功转换为 YAML")
