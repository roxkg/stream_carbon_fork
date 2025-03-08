import numpy as np
import matplotlib.pyplot as plt



# 示例数据
categories = ['1', '3', '5', '7', '8']  # 横轴类别 (Num Apps)
num_categories = len(categories)

# 生成数据（示例）
fpga_oc = np.array([10, 20, 30, 40, 50]) * 1e6  # FPGA Operational Carbon
fpga_ec = np.array([15, 25, 35, 45, 55]) * 1e6  # FPGA Embodied Carbon
asic_oc = np.array([5, 10, 15, 20, 25]) * 1e6   # ASIC Operational Carbon

# 另一组数据 (App Lifetime 对应的另一组数据)
fpga_oc_2 = np.array([35, 30, 25, 40, 45]) * 1e6
fpga_ec_2 = np.array([20, 25, 30, 35, 40]) * 1e6
asic_oc_2 = np.array([8, 12, 18, 22, 28]) * 1e6

# 设定柱状图的宽度
bar_width = 0.3  
x = np.arange(num_categories)  # X 轴刻度位置

# 创建图表
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制左边柱状图 (Num Apps)
ax.bar(x - bar_width/2, fpga_ec, width=bar_width, label="FPGA EC", color="purple", edgecolor ='grey')
ax.bar(x - bar_width/2, fpga_oc, width=bar_width, bottom=fpga_ec, label="FPGA OC", color="gold",edgecolor ='grey')
ax.bar(x - bar_width/2, asic_oc, width=bar_width, bottom=fpga_ec + fpga_oc, label="ASIC OC", color="lightgreen", edgecolor ='grey')

# 绘制右边柱状图 (App Lifetime)
ax.bar(x + bar_width/2, fpga_ec_2, width=bar_width, color="purple", edgecolor ='grey')
ax.bar(x + bar_width/2, fpga_oc_2, width=bar_width, bottom=fpga_ec_2, color="gold", edgecolor ='grey')
ax.bar(x + bar_width/2, asic_oc_2, width=bar_width, bottom=fpga_ec_2 + fpga_oc_2, color="lightgreen", edgecolor ='grey')

# 设置 X 轴
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_xlabel("Num Apps / App Lifetime")
ax.set_ylabel("Eq. Kgs of CO₂")
ax.set_yscale("linear")  # 也可以尝试对数刻度 `log`

# 添加图例
ax.legend()

# 显示图表
plt.show()
