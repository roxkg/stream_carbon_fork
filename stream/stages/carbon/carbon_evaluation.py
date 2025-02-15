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
    # mem_area = scme.mem_area
    op_CI = scme.carbon_param.CI_op
    energy_value = float(scme.energy)
    op_co2 = op_CI * energy_value / (3.6 * 10**18)
    lifespan = 2
    taskspan = scme.latency/(1.8*10**9)   #clock freq = 1.8GHz
    lifeop_co2 = lifespan*365*24*60*60*0.2/taskspan * op_co2
    
    
    # current technology node: 16nm -> approximated by 14nm 
    # all data from ECO-CHIP
    defect_density = 0.09
    # one large chip
    cpa = 16.84 
    rdl_layers = 6
    area = (12000/1024*total_area)*0.000008*0.000008
    print(area)
    wastage_extra_cfp = Si_wastage_accurate_t(wafer_dia=450,chip_area=area,cpa_factors=cpa)
    yields = yield_calc(area,defect_density)
    mfg_carbon = cpa*area / yields
    mfg_carbon = mfg_carbon+wastage_extra_cfp

    package_defect_den = defect_density/4
    pacakge_yield = yield_calc(area,package_defect_den)
    wastage_extra_cfp = Si_wastage_accurate_t(wafer_dia=450,chip_area=area,cpa_factors=cpa)
    mfg_carbon = cpa*area / pacakge_yield
    hi_carbon = mfg_carbon+wastage_extra_cfp

    package_carbon = hi_carbon* 0.4916
    package_carbon *= rdl_layers/8
    cemb = package_carbon + mfg_carbon

    with open(output_path, "w") as file: 
        file.write(f"op_co2 = {op_co2}g\ntotal life co2 = {lifeop_co2}\n area={total_area}\n")
        file.write(f"em_co2 = {cemb}")
    # RDL fan-out 
    # "rdl_layers" = 6, "defect_density" = 0.09 

def yield_calc(area, defect_density):
    yield_val = (1+(defect_density*1e4)*(area*1e-6)/10)**-10
    return yield_val

def Si_wastage_accurate_t(wafer_dia,chip_area,cpa_factors):
    si_area = (math.pi * (wafer_dia ** 2))/4
    dpw = math.pi * wafer_dia * ((wafer_dia/(4*chip_area)) - (1/math.sqrt(2*chip_area)))
    area_wastage = si_area - (math.floor(dpw)*chip_area)
    #unused_si_cfp = wastage_cfp(area_wastage,techs,scaling_factors)
    unused_si_cfp = area_wastage*cpa_factors
    wastage_si_cfp_pdie = unused_si_cfp/dpw
    return wastage_si_cfp_pdie

"""
def EMIB_carbon(

): 
    #0.33 in 16 convert to 7nm
    router_area = 0.33/np.array([scaling_factors[ty].loc[14, 'area'] for ty in types])
    emib_area =  [5*5]*num_if
    #print("NUMBER OF INTERFACES",num_if)
    emib_carbon, _, _ = Si_chip([22]*num_if, ["logic"]*num_if, emib_area, scaling_factors,transistors_per_gate,
                        power_per_core,carbon_per_kWh,True)
    package_carbon = np.sum(emib_carbon)* scaling_factors['beolVfeol'].loc[22,'beolVfeol']
"""





    
    
    
    


     