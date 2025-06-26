import logging as _logging
import re

from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.stages.carbon.carbon_evaluation_multicore import calculate_carbon
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--is_chiplet", action="store_true", default=False)
parser.add_argument("--interposer_area", type=float, default=0.0)
parser.add_argument("--rcy_mat_frac", type=float, default=0.0)
parser.add_argument("--accelerator", type=str, default="stream/inputs/examples/hardware/carbon/simba.yaml")
args = parser.parse_args()
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
accelerator = args.accelerator
# accelerator = "stream/inputs/examples/hardware/carbon/simba.yaml"
carbon_path = "stream/inputs/examples/carbon/simba.yaml"
workload_path = "stream/inputs/examples/workload/resnet50.onnx"
mapping_path = "stream/inputs/examples/mapping/simba.yaml"
mode = "fused"
direction = "simba_16core"
is_chiplet = args.is_chiplet
interposer_area = args.interposer_area
# interposer_area = 2271
rcy_mat_frac = args.rcy_mat_frac
# rcy_mat_frac = 0.9
rcy_cpa_frac = 0.4
is_chiplet = True
# interposer_area = 2441
# interposer_area = 2256 # 4.7*3 + 25.4951*2)*(4.7*2 + 25.4951) # 1463.22
# interposer_area = (4.7*7 + 3.15*6)*(4.7*7 + 3.15*6) # 1463.22
layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
nb_ga_generations = 4
nb_ga_individuals = 4
##############################################################################################

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-{direction}-{is_chiplet}-{interposer_area}-{rcy_mat_frac}-genetic_algorithm"
######################################################################

##############PLOTTING###############
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#####################################


################################PATHS################################
timeline_fig_path_plotly = f"outputs/{experiment_id}/schedule.html"
memory_fig_path = f"outputs/{experiment_id}/memory.png"
json_path = f"outputs/{experiment_id}/scme.json"
#####################################################################

scme = optimize_allocation_ga(
    hardware=accelerator,
    workload=workload_path,
    carbon=carbon_path,
    interposer_area=interposer_area,
    rcy_mat_frac=rcy_mat_frac,
    rcy_cpa_frac=rcy_cpa_frac,
    opt="tCDP",
    is_chiplet= is_chiplet,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    nb_ga_generations=nb_ga_generations,
    nb_ga_individuals=nb_ga_individuals,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=True,
)

# Load in the CostModelEvaluationLUT from the run
cost_lut_path = f"outputs/{experiment_id}/cost_lut.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)

# Plotting schedule timeline of best SCME
visualize_timeline_plotly(
    scme,
    draw_dependencies=draw_dependencies,
    draw_communication=plot_data_transfer,
    fig_path=timeline_fig_path_plotly,
    cost_lut=cost_lut,
)
# calculate_carbon(scme, False)
# calculate_carbon(scme, True)

f = open(f"outputs/{experiment_id}/metrics.txt", "w")
f.write(f"total delay = {scme.latency} cycles\ntotal energy = {scme.energy} pJ\nembodied carbon = {scme.embodied_carbon}\noperational carbon = {scme.carbon}\ntCDP = {scme.tCDP}\nCD={scme.CD}\nED={scme.ED}\n")
f.close()


print(f"total delay = {scme.latency} cycles\ntotal energy = {scme.energy} pJ\nembodied carbon = {scme.embodied_carbon}\noperational carbon = {scme.carbon}\ntCDP = {scme.tCDP}\nCD={scme.CD}\nED={scme.ED}\n")

# Plotting memory usage of best SCME
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)

# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)



