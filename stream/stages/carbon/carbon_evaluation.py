import numpy as np
import pandas as pd
import argparse
import logging
import pickle
import math
from typing import TYPE_CHECKING

from stream.stages.stage import Stage, StageCallable
from zigzag.utils import open_yaml

if TYPE_CHECKING:
    from stream.cost_model.cost_model import StreamCostModelEvaluation
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)

def calculate_carbon(
    scme: "StreamCostModelEvaluation",
    output_path: str = "outputs/carbon.txt",
):
    #area is in unit: normalized by 1 MAC area for 1 element calculation
    total_area = scme.area_total
    mem_area = scme.mem_area
    op_CI = scme.carbon_param.CI_op
    energy_value = float(scme.energy)
    # pJ -> kWh
    op_co2 = op_CI * energy_value / (3.6 * 10**18)
    lifespan = scme.carbon_param.lifetime
    frequency = scme.carbon_param.frequency
    taskspan = scme.latency/(frequency*10**9)   
    # year -> s 
    # lifetime working proportion: 0.2
    lifeop_co2 = lifespan*365*24*60*60*0.2/taskspan * op_co2
    
    # current technology node: 16nm -> approximated by 14nm 
    # all data from ECO-CHIP
    #defect_density = 0.09
    defect_density = get_defect_rate(scme.carbon_param.technology_node)
    # one large chip
    cpa = get_carbon_per_area(scme.carbon_param.technology_node) 
    
    
    area = (12000/1024*total_area)*0.000008*0.000008
    print(area)
    wastage_extra_cfp = waste_carbon_per_die(wafer_dia=450,chip_area=area,cpa_factors=cpa)
    yields = yield_calc(area,defect_density)
    mfg_carbon = cpa*area / yields
    mfg_carbon = mfg_carbon+wastage_extra_cfp

    """
    # below shoule be package carbon emission
    rdl_layers = 6
    package_defect_den = defect_density/4
    pacakge_yield = yield_calc(area,package_defect_den)
    wastage_extra_cfp = waste_carbon_per_die(wafer_dia=450,chip_area=area,cpa_factors=cpa)
    mfg_carbon = cpa*area / pacakge_yield
    hi_carbon = mfg_carbon+wastage_extra_cfp
    package_carbon = hi_carbon* 0.4916
    package_carbon *= rdl_layers/8
    cemb = package_carbon + mfg_carbon
    """
    cemb = mfg_carbon
    
    with open(output_path, "w") as file: 
        file.write(f"op_co2 = {op_co2}g\ntotal life co2 = {lifeop_co2}\n area={total_area}\n")
        file.write(f"em_co2 = {cemb}")
    # RDL fan-out 
    # "rdl_layers" = 6, "defect_density" = 0.09 

def yield_calc(area, defect_density):
    yield_val = (1+(defect_density*1e4)*(area*1e-6)/10)**-10
    return yield_val

def waste_carbon_per_die(diameter,chip_area,cpa_factors):
    wafer_area = (math.pi * (diameter ** 2))/4
    dies_per_wafer = math.pi * diameter * ((diameter/(4*chip_area)) - (1/math.sqrt(2*chip_area)))
    area_wastage = wafer_area - (math.floor(dies_per_wafer)*chip_area)
    #unused_si_cfp = wastage_cfp(area_wastage,techs,scaling_factors)
    unused_si_cfp = area_wastage*cpa_factors
    wastage_si_cfp_pdie = unused_si_cfp/dies_per_wafer
    return wastage_si_cfp_pdie

def get_defect_rate(technology_node):
    defect_density = open_yaml("stream/inputs/examples/carbon/defect_density.yaml")
    nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
    # in node is in 7,10,14,22,28, return defect density directly
    if technology_node in nodes:
        return defect_density[nodes.index(technology_node)]

    # or use interpolation
    defect_rate = np.interp(technology_node, nodes, defect_density)

    return defect_rate

def get_carbon_per_area(technology_node):
    cpas = open_yaml("stream/inputs/examples/carbon/carbon_per_area.yaml")
    nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
    # in node is in 7,10,14,22,28, return defect density directly
    if technology_node in nodes:
        return cpas[nodes.index(technology_node)]

    # or use interpolation
    carbon_per_area = np.interp(technology_node, nodes, cpas)

    return carbon_per_area





    
    
    
    


     