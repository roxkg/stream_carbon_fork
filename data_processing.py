import os
import re

# 设置根目录路径（请修改为你的实际路径）
# root_dir = 'C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/simba/9core'  # 例如：'/home/user/simba-experiments'

def extract_numeric_key(name):
    match = re.search(r'-([0-9]+\.[0-9]+)-0\.0-genetic_algorithm', name)
    if match:
        return float(match.group(1))
    else:
        return float('inf')  # 匹配失败的放最后

def get_tcdp_value(root_dir):
    # 存储提取到的 tCDP 数值
    tcdp_values = []

    # 获取并排序所有文件夹
    folders = sorted(
    [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))],
    key=extract_numeric_key
)
    # folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    # print(f"Found folders: {folders}")

    # 正则表达式用于提取 tCDP 的值（支持小数、整数、科学记数法）
    float_pattern = r'tCDP\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'

    # 遍历文件夹并提取 tCDP 值
    for folder in folders:
        metrics_path = os.path.join(root_dir, folder, 'metrics.txt')
        try:
            with open(metrics_path, 'r') as f:
                content = f.read()
                match = re.search(float_pattern, content)
                if match:
                    tcdp_values.append(float(match.group(1)))
                else:
                    print(f"[警告] 未找到 tCDP 于文件: {metrics_path}")
        except FileNotFoundError:
            print(f"[错误] 文件不存在: {metrics_path}")
    return tcdp_values

def get_cd_value(root_dir):
    # 存储提取到的 tCDP 数值
    tcdp_values = []

    # 获取并排序所有文件夹
    folders = sorted(
    [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))],
    key=extract_numeric_key
)
    # folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    # print(f"Found folders: {folders}")

    # 正则表达式用于提取 tCDP 的值（支持小数、整数、科学记数法）
    float_pattern = r'CD\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'

    # 遍历文件夹并提取 tCDP 值
    for folder in folders:
        metrics_path = os.path.join(root_dir, folder, 'metrics.txt')
        try:
            with open(metrics_path, 'r') as f:
                content = f.read()
                match = re.search(float_pattern, content)
                if match:
                    tcdp_values.append(float(match.group(1)))
                else:
                    print(f"[警告] 未找到 tCDP 于文件: {metrics_path}")
        except FileNotFoundError:
            print(f"[错误] 文件不存在: {metrics_path}")
    return tcdp_values
