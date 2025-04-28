import subprocess

interspaces=[0,1,2,3,4,5,6,6.5]
interposer_areas=[(interspace * 5 + 3.0146*6)**2 for interspace in interspaces]

recycle_ratios = [0,0.2,0.6,0.8]

# # 再跑 is_chiplet=False（不带 flag）
# print("Running with is_chiplet=False...")
# subprocess.run(["python", "main_stream_ga.py"], check=True)

# # 先跑 is_chiplet=True（带 flag）
# print("Running with is_chiplet=True...")
# for area in interposer_areas:
#     subprocess.run(["python", "main_stream_ga.py", "--interposer_area", str(area),"--is_chiplet"], check=True)


for recycle_ratio in recycle_ratios:
    subprocess.run(["python", "main_stream_ga.py","--rcy_mat_frac", str(recycle_ratio)], check=True)
    subprocess.run(["python", "main_stream_ga.py", "--rcy_mat_frac", str(recycle_ratio),"--is_chiplet"], check=True)
