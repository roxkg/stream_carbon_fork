import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 创建示例数据
categories = ["1", "2"]  # X 轴的类别
labels_asic1 = ["Ope", "Mfg", "Des"]  # ASIC1 组成部分
labels_asic2 = ["Des", "Mfg", "Eol","App-Dev"]  # ASIC2 组成部分

values_asic1 = [160204952.8593164,22937670.693933833,5379.677266371265]  # ASIC1 各部分数据
values_asic2 = [5379.677266371265, 22937670.693933833, 780.0, 50.4]  # ASIC2 各部分数据

# 设定颜色（匹配示例图片）
colors_asic1 = ["blue", "orange", "green"]  # ASIC1 部分颜色
colors_asic2 = ["green", "orange","red","yellow"]  # ASIC2 部分颜色

# 创建画布
fig, ax = plt.subplots(figsize=(8, 6))
ax2 = ax.twinx()
# **绘制 ASIC1**
legend_handles = {}
bottom_asic1 = 0  # 初始堆叠底部
x1 = 0  # ASIC1 的 X 位置
for i, val in enumerate(values_asic1):
    bar = ax.bar(x1, val, color=colors_asic1[i], edgecolor="black", width=0.6, bottom=bottom_asic1, label=labels_asic1[i] if i == 0 else "")
    bottom_asic1 += val  # 更新底部位置
    legend_handles[labels_asic1[i]] = bar[0]

# **绘制 ASIC2**
bottom_asic2 = 0  # 初始堆叠底部
x2 = 1  # ASIC2 的 X 位置
for i, val in enumerate(values_asic2):
    bar = ax2.bar(x2, val, color=colors_asic2[i], edgecolor="black", width=0.6, bottom=bottom_asic2, label=labels_asic2[i] if i == 0 else "")
    bottom_asic2 += val  # 更新底部位置
    legend_handles[labels_asic2[i]] = bar[0]

# **设置 X 轴和 Y 轴标签**
ax.set_ylabel("Eq. kgs of CO₂", fontsize=12)
ax.set_xlabel("")
ax.set_xticks([x1, x2])
ax.set_xticklabels(categories, fontsize=12)

# **自定义图例**
ax.legend(legend_handles.values(), legend_handles.keys(), loc="upper left", fontsize=10, frameon=True)
ax2.set_ylim(0, min(bottom_asic1, bottom_asic2) * 1.2)
# 显示图表
plt.show()
