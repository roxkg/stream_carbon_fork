from matplotlib import pyplot as plt
from data_processing import get_tcdp_value

tcdp_values1 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/simba/9core')
tcdp_values2 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/simba/12core')

tcdp_values3 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/simba/36core')
tcdp_values4 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/simba/16core')
threshold1 = 4.444205510856269e-09
# ecochip1 = 5.85625e-09
x_values = [round(x * 0.1, 1) for x in range(len(tcdp_values1))]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_values, tcdp_values1, marker='o', label='9chiplet')
plt.plot(x_values, tcdp_values2, marker='^', label='12chiplet')
plt.plot(x_values, tcdp_values4, marker='x', label='16chiplet')
plt.plot(x_values, tcdp_values3, marker='s', label='36chiplet')
plt.axhline(y=threshold1, color='red', linestyle='--', linewidth=1.5, label=f'Reference tCDP = {threshold1:.2e}')
# plt.axhline(y=ecochip1, color='blue', linestyle='--', linewidth=1.5, label=f'Ecochip tCDP = {ecochip1:.2e}')
plt.xlabel("Recycling Ratio", fontweight='bold')
plt.ylabel("tCDP(g*s)", fontweight='bold')
plt.legend()
plt.grid(True)


# samba_tcdp1 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/sambanova/2core')
# samba_tcdp1 =[x+0.15e-6 for x in samba_tcdp1[1:]]
# samba_tcdp1.insert(0, 5.649711393054508e-07)
# samba_tcdp2 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/sambanova/4core')
# samba_tcdp2 =[x+0.15e-6 for x in samba_tcdp2[1:]]
# samba_tcdp2.insert(0, 6.959711393054508e-07)
# samba_tcdp3 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/sambanova/16core')
# samba_tcdp3[0] = 1.81e-6
# samba_tcdp4 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/sambanova/12core')
# samba_tcdp4[0] = 1.07e-6
# threshold2 = 9.284828907634065e-07
# # ecochip2 = 3.32112e-07
# plt.subplot(1, 2, 2)
# plt.plot(x_values, samba_tcdp1, marker='o', label='2chiplet')
# plt.plot(x_values, samba_tcdp2, marker='^', label='4chiplet')
# plt.plot(x_values, samba_tcdp4, marker='x', label='12chiplet')
# plt.plot(x_values, samba_tcdp3, marker='s', label='16chiplet')
# plt.axhline(y=threshold2, color='red', linestyle='--', linewidth=1.5, label=f'Reference tCDP = {threshold2:.2e}')
# # plt.axhline(y=ecochip2, color='blue', linestyle='--', linewidth=1.5, label=f'Ecochip tCDP = {ecochip2:.2e}')
# plt.xlabel("Recycling Ratio", fontweight='bold')
# plt.ylabel("tCDP(g*s)", fontweight='bold')
# plt.grid(True)
# plt.legend()
plt.show()