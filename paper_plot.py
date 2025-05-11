import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# simba: 
# chiplet
core_noc_mfg = 102.99883884      # g/chiplet 
pck_mfg = 13664.60468946         # g
core_des = 14.751745348731003    # g/chiplet
noc_des = 7.827456715653184      # g/chiplet 
noc_area = 2.08
core_area = 3.92
frequency = 1.8  

latency = 6608018
energy = 1121399991.6000018

lifespan = 17520*0.2
taskspan = latency/(frequency*10**9)/3600    # hours
energy = energy/(10**12)/3600000        # kWh

core_mfg_CD = core_noc_mfg *core_area / (core_area + noc_area) * 36 /lifespan * taskspan * taskspan * 3600 
noc_mfg_CD = core_noc_mfg *noc_area / (core_area + noc_area) * 36 /lifespan * taskspan * taskspan * 3600
pck_mfg_CD = pck_mfg /lifespan * taskspan * taskspan * 3600

core_des_CD = core_des /lifespan * taskspan * taskspan * 3600
noc_des_CD = noc_des /lifespan * taskspan * taskspan * 3600
pkg_des_CD = 0

ED = 820*energy*taskspan * 3600

# SoC
soc_mfg = 2876.19650982         # g
pkg_mfg = 0
noc_mfg = 0
soc_des = 531.0628325543162      # g
noc_des = 0

latency = 6544040
energy  = 1055474769.2000018

lifespan = 17520*0.2
taskspan = latency/(frequency*10**9)/3600    # hours
energy = energy/(10**12)/3600000        # kWh

soc_mfg_CD = soc_mfg /lifespan * taskspan * taskspan * 3600 
soc_noc_mfg_CD = 0
soc_pck_mfg_CD = 0

soc_des_CD = soc_des /lifespan * taskspan * taskspan * 3600
soc_noc_des_CD = 0
soc_pkg_des_CD = 0

soc_ED = 820*energy*taskspan * 3600

# sambanova:
# chiplet
core_noc_mfg_2 = 60935.15310412      # g/chiplet 
pck_mfg_2 = 22719.71814717         # g
core_des_2 = 11255.702502909282    # g/chiplet
noc_des_2 = 1670.4576152727557      # g/chiplet 
noc_area_2 = 84
core_area_2 = 566
frequency_2 = 1.8  

latency_2 = 9742820
energy_2 = 153654068721.7996

lifespan_2 = 17520*0.2
taskspan_2 = latency_2/(frequency_2*10**9)/3600    # hours
energy_2 = energy_2/(10**12)/3600000        # kWh

core_mfg_CD_2 = core_noc_mfg_2 *core_area_2 / (core_area_2 + noc_area_2) * 2 /lifespan_2 * taskspan_2 * taskspan_2 * 3600 
noc_mfg_CD_2 = core_noc_mfg_2 *noc_area_2 / (core_area_2 + noc_area_2) * 2 /lifespan_2 * taskspan_2 * taskspan_2 * 3600
pck_mfg_CD_2 = pck_mfg_2 /lifespan_2 * taskspan_2 * taskspan_2 * 3600

core_des_CD_2 = core_des_2 /lifespan_2 * taskspan_2 * taskspan_2 * 3600
noc_des_CD_2 = noc_des_2 /lifespan_2 * taskspan_2 * taskspan_2 * 3600
pkg_des_CD_2 = 0

ED_2 = 820*energy_2*taskspan_2 * 3600

# SoC
soc_mfg_2 = 235813.38888404         # g
pkg_mfg_2 = 0
noc_mfg_2 = 0
soc_des_2 = 22511      # g
noc_des_2 = 0

latency_2 = 10679260
energy_2  = 153654097313.3996

lifespan_2 = 17520*0.2
taskspan_2 = latency_2/(frequency_2*10**9)/3600    # hours
energy_2 = energy_2/(10**12)/3600000        # kWh

soc_mfg_CD_2 = soc_mfg_2 /lifespan_2 * taskspan_2 * taskspan_2 * 3600 
soc_noc_mfg_CD_2 = 0
soc_pck_mfg_CD_2 = 0

soc_des_CD_2 = soc_des_2 /lifespan_2 * taskspan_2 * taskspan_2 * 3600
soc_noc_des_CD_2 = 0
soc_pkg_des_CD_2 = 0

soc_ED_2 = 820*energy_2*taskspan_2 * 3600


bar_labels = [['chiplet', 'SoC'], ['chiplet', 'SoC']]
colors = ['skyblue', 'salmon', 'lightgreen']
hatches = ['//', 'xx', '.']
part_labels = ['core', 'noc', 'pck']

# 每个柱状图由两种颜色组成，每种颜色再分三段
data_all = [
    [
    # Bar 1: [mfg: core, noc, pkg], [des: core, noc, pkg], ED
    [[core_mfg_CD, noc_mfg_CD, pck_mfg_CD], [core_des_CD, noc_des_CD, pkg_des_CD], ED],   
    # Bar 2: [mfg: core, noc, mfg], [des: core, noc, mfg], ED
    [[soc_mfg_CD, soc_noc_mfg_CD, soc_pck_mfg_CD], [soc_des_CD, soc_noc_des_CD, soc_pkg_des_CD], soc_ED]],
    # Bar 3: [mfg: core, noc, pkg], [des: core, noc, pkg], ED
    [
    [[core_mfg_CD_2, noc_mfg_CD_2, pck_mfg_CD_2], [core_des_CD_2, noc_des_CD_2, pkg_des_CD_2], ED_2],   
    # Bar 4: [mfg: core, noc, mfg], [des: core, noc, mfg], ED
    [[soc_mfg_CD_2, soc_noc_mfg_CD_2, soc_pck_mfg_CD_2], [soc_des_CD_2, soc_noc_des_CD_2, soc_pkg_des_CD_2], soc_ED_2]]  
]    

bar_width = 0.2
group_gap = 1.0
bar_gap = 0.2

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

for ax_idx, (ax, data) in enumerate(zip(axes, data_all)):
    x = np.arange(len(data))  # 仅两根柱子：x = [0, 1]
    x = [0, 0.5]
    for i, (bar_x, bar_data) in enumerate(zip(x, data)):
        bottom = 0
        # 第一种颜色，三段图案
        for j, val in enumerate(bar_data[0]):
            ax.bar(bar_x, val, width=bar_width, bottom=bottom,
                   color=colors[0], hatch=hatches[j], edgecolor='black')
            bottom += val
        # 第二种颜色，三段图案
        for j, val in enumerate(bar_data[1]):
            ax.bar(bar_x, val, width=bar_width, bottom=bottom,
                   color=colors[1], hatch=hatches[j], edgecolor='black')
            bottom += val
        # 第三种颜色，无图案
        ax.bar(bar_x, bar_data[2], width=bar_width, bottom=bottom,
               color=colors[2], edgecolor='black')

    # 设置标签
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels[ax_idx])
    ax.set_ylabel('tCDP(g*s)')
    if ax_idx == 0:
        ax.set_title('SIMBA')
    else:
        ax.set_title('SambaNova')

legend_elements = []
for color_idx in range(2):
    for hatch_idx, hatch in enumerate(hatches):
        label = f"{['Cmfg*D', 'Cdes*D'][color_idx]} - {part_labels[hatch_idx]}"
        legend_elements.append(Patch(facecolor=colors[color_idx], hatch=hatch,
                                     edgecolor='black', label=label))
legend_elements.append(Patch(facecolor=colors[2], edgecolor='black', label='ED'))

# ✅ 图例放在右侧，center left 相对于整个 figure
fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.6, 0.8))

print("simba:\n" \
      "chiplet:\n" \
      "core_mfg_CD: ", core_mfg_CD, "\n" \
      "noc_mfg_CD: ", noc_mfg_CD, "\n" \
      "pck_mfg_CD: ", pck_mfg_CD, "\n" \
      "core_des_CD: ", core_des_CD, "\n" \
      "noc_des_CD: ", noc_des_CD, "\n" \
      "pkg_des_CD: ", pkg_des_CD, "\n" \
      "ED: ", ED, "\n"  \
      "-------------------\n" \
      "soc:\n" \
      "soc_mfg_CD: ", soc_mfg_CD, "\n" \
      "soc_noc_mfg_CD: ", soc_noc_mfg_CD, "\n" \
      "soc_pck_mfg_CD: ", soc_pck_mfg_CD, "\n" \
      "soc_des_CD: ", soc_des_CD, "\n" \
      "soc_noc_des_CD: ", soc_noc_des_CD, "\n" \
      "soc_pkg_des_CD: ", soc_pkg_des_CD, "\n" \
      "ED: ", soc_ED, "\n" )

print("--------------------------------------------------")
print("samba:\n" \
"chiplet:\n" \
"core_mfg_CD: ", core_mfg_CD_2, "\n" \
"noc_mfg_CD: ", noc_mfg_CD_2, "\n" \
"pck_mfg_CD: ", pck_mfg_CD_2, "\n" \
"core_des_CD: ", core_des_CD_2, "\n" \
"noc_des_CD: ", noc_des_CD_2, "\n" \
"pck_des_CD: ", pkg_des_CD_2, "\n" \
"ED: ", ED_2, "\n"  \
"soc:\n" \
"soc_mfg_CD: ", soc_mfg_CD_2, "\n" \
"soc_noc_mfg_CD: ", soc_noc_mfg_CD_2, "\n" \
"soc_pck_mfg_CD: ", soc_pck_mfg_CD_2, "\n" \
    "soc_des_CD: ", soc_des_CD_2, "\n" \
"soc_noc_des_CD: ", soc_noc_des_CD_2, "\n" \
    "soc_pkg_des_CD: ", soc_pkg_des_CD_2, "\n" \
"ED: ", soc_ED_2, "\n" )

plt.show()

