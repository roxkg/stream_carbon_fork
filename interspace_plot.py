from matplotlib import pyplot as plt
from data_processing import get_tcdp_value

tcdp_values1 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/simba/9core')
tcdp_values2 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/simba/12core')
tcdp_values3 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/simba/36core')
tcdp_values4 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/simba/16core')
tcdp_values5 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/simba/4core')
tcdp_values6 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/simba/6core')
# breakpoint()
threshold1 = 4.444205510856269e-09
# ecochip1 = 5.85625e-09
x_values = [1, 2, 3, 4, 4.7, 6]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_values, tcdp_values5, marker='.', label='4chiplet')
plt.plot(x_values, tcdp_values6, marker='*', label='6chiplet')
plt.plot(x_values, tcdp_values1, marker='o', label='9chiplet')
plt.plot(x_values, tcdp_values2, marker='^', label='12chiplet')
plt.plot(x_values, tcdp_values3[1:], marker='s', label='36chiplet')
plt.plot(x_values, tcdp_values4, marker='x', label='16chiplet')
plt.axhline(y=threshold1, color='red', linestyle='--', linewidth=1.5, label=f'SoC tCDP = {threshold1:.2e}')
# plt.axhline(y=ecochip1, color='blue', linestyle='--', linewidth=1.5, label=f'Ecochip tCDP = {ecochip1:.2e}')
plt.xlabel("Interspace", fontweight='bold')
plt.ylabel("tCDP(g*s)", fontweight='bold')
plt.legend()
plt.grid(True)


samba_tcdp1 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/sambanova/2core')
samba_tcdp1 = [ 5.08549146305996e-07, 5.325044636007652e-07, 5.44764089880198e-07, 5.847667124856997e-07, 5.563256362150821e-07, 6.317415466281949e-07]
samba_tcdp2 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/sambanova/9core')
samba_tcdp3 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/sambanova/16core')
samba_tcdp4 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/sambanova/12core')
samba_tcdp5 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/sambanova/4core')
samba_tcdp6 = get_tcdp_value('C:/Users/mg011/Desktop/新建文件夹/interposer_fig_data/sambanova/6core')
threshold2 = 9.284828907634065e-07
# ecochip2 = 3.32112e-07
plt.subplot(1, 2, 2)
plt.plot(x_values, samba_tcdp1, marker='o', label='2chiplet')
plt.plot(x_values, samba_tcdp5, marker='.', label='4chiplet')
plt.plot(x_values, samba_tcdp6, marker='*', label='6chiplet')
plt.plot(x_values, samba_tcdp2, marker='^', label='9chiplet')
plt.plot(x_values, samba_tcdp3, marker='s', label='16chiplet')
plt.plot(x_values, samba_tcdp4, marker='x', label='12chiplet')
plt.axhline(y=threshold2, color='red', linestyle='--', linewidth=1.5, label=f'SoC tCDP = {threshold2:.2e}')
# plt.axhline(y=ecochip2, color='blue', linestyle='--', linewidth=1.5, label=f'Ecochip tCDP = {ecochip2:.2e}')
plt.xlabel("Interspace", fontweight='bold')
plt.ylabel("tCDP(g*s)", fontweight='bold')
plt.grid(True)
plt.legend()
plt.show()