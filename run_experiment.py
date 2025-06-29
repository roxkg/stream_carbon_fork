from math import sqrt
import subprocess
# interspaces=[1, 2, 3, 4, 4.7, 6]
# interposer_areas=[(interspace * 5 + 3.3015*4)*(interspace * 5 + 3.3015*4) for interspace in interspaces]
# simba = "stream/inputs/examples/hardware/carbon/simba.yaml"
# for interposer_area in interposer_areas:
#     subprocess.run(["python", "main_stream_ga.py", "--interposer_area", str(interposer_area),"--accelerator", simba,"--is_chiplet"], check=True)
# interposer_areas=[(interspace * 5 + 3.7202*4)*(interspace*4 + 3.7202*3) for interspace in interspaces]
# interposer_areas = [1463.22, 1829.52]

# times = [0.4, 0.6,0.8,1,1.2]
# ds = [sqrt(566*time*2/4+84) for time in times]
# interposer_areas = [(4.7 * 3 + d*2)*(4.7 * 3 + d*2) for d in ds]
# times = [ 1.2, 1.4, 1.6, 1.8]
# ds = [sqrt(3.92*time*36/4+2.08) for time in times]
# interposer_areas = [(4.7 * 3 + d*2)*(4.7 * 3 + d*2) for d in ds]
recycle_ratios = [0, 0.1, 0.2, ]
"""
"stream/inputs/examples/hardware/carbon/sambanova_04.yaml",
                "stream/inputs/examples/hardware/carbon/sambanova_06.yaml", 
                "stream/inputs/examples/hardware/carbon/sambanova_08.yaml",
                "stream/inputs/examples/hardware/carbon/sambanova.yaml",
                "stream/inputs/examples/hardware/carbon/sambanova_12.yaml",
"stream/inputs/examples/hardware/carbon/simba_0875area.yaml",
                "stream/inputs/examples/hardware/carbon/simba.yaml",
                "stream/inputs/examples/hardware/carbon/simba_12area.yaml",
                "stream/inputs/examples/hardware/carbon/simba_14.yaml",
                "stream/inputs/examples/hardware/carbon/simba_16.yaml",
                "stream/inputs/examples/hardware/carbon/simba_18.yaml",
                "stream/inputs/examples/hardware/carbon/simba_2area.yaml",

                
                
                
"""
# # 再跑 is_chiplet=False（不带 flag）
# print("Running with is_chiplet=False...")
# subprocess.run(["python", "main_stream_ga.py"], check=True)
accelerators = "stream/inputs/examples/hardware/carbon/simba.yaml"
# times = [ 2.0]
# ds = [sqrt(3.92*36/16*time+2.08) for time in times]
# interposer_areas = [(4.7 * 5 + d*4)*(4.7 * 5 + d*4) for d in ds]
interposer_area = (4.7*5 + 3.3015*4)*(4.7*5 + 3.3015*4)
# breakpoint()
# # 先跑 is_chiplet=True（带 flag）
# interposer_area = 1463.221
# print("Running with is_chiplet=True...")
# for i in range(len(interposer_areas)):
#     subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerators[i],"--interposer_area", str(interposer_areas[i]),"--is_chiplet"], check=True)

# for i in range(len(accelerators)):
    # print(f"Running for {accelerator}...")
    # subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerators[i],"--interposer_area", str(interposer_areas[i]),"--is_chiplet"], check=True)
# interposer_area = 2747.278
for recycle_ratio in recycle_ratios:
    subprocess.run(["python", "main_stream_ga.py","--rcy_mat_frac", str(recycle_ratio),"--is_chiplet","--accelerator", accelerators,"--interposer_area", str(interposer_area)], check=True)
    # subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerators[0], "--interposer_area", str(interposer_areas),"--rcy_mat_frac", str(recycle_ratio),"--is_chiplet"], check=True)
# x = 54644
# y = [((1-ratio)*x +(ratio*0.4*x)+19644.88)/1000 for ratio in recycle_ratios]
# ref = [29.228, 27.548, ]
# CD = get_cd_value("C:/Users/mg011/Desktop/新建文件夹/recycle_fig_data/sambanova/2core")
# print(y)
# result = []

# accelerator = "stream/inputs/examples/hardware/carbon/simba_16.yaml"
# interposer_area = (7*4.7 + 6*2.88998)*(7*4.7 + 6*2.88998)
# subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerator,"--interposer_area", str(interposer_area),"--is_chiplet"], check=True)
# subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerator,"--interposer_area", str(interposer_area)], check=True)

# accelerator = "stream/inputs/examples/hardware/carbon/sambanova_08.yaml"
# interposer_area = (3*4.7 + 2*23.169)*(2*4.7 + 23.169)
# subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerator,"--interposer_area", str(interposer_area),"--is_chiplet"], check=True)
# subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerator,"--interposer_area", str(interposer_area)], check=True)

# accelerator = "stream/inputs/examples/hardware/carbon/sambanova.yaml"
# interposer_area = (3*4.7 + 2*17.618)*(2*4.7 + 17.618)
# subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerator,"--interposer_area", str(interposer_area),"--is_chiplet"], check=True)
# subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerator,"--interposer_area", str(interposer_area)], check=True)

# accelerator = "stream/inputs/examples/hardware/carbon/sambanova.yaml"
# interposer_area = (3*4.7 + 2*27.626)*(2*4.7 + 27.626)
# # subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerator,"--interposer_area", str(interposer_area),"--is_chiplet"], check=True)
# subprocess.run(["python", "main_stream_ga.py", "--accelerator", accelerator], check=True)
