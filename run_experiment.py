import subprocess

# 先跑 is_chiplet=True（带 flag）
print("Running with is_chiplet=True...")
subprocess.run(["python", "main_stream_ga.py", "--is_chiplet"], check=True)

# 再跑 is_chiplet=False（不带 flag）
print("Running with is_chiplet=False...")
subprocess.run(["python", "main_stream_ga.py"], check=True)
