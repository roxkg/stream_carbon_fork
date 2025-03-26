import numpy as np
import pandas as pd
import argparse
import logging
import pickle
import math
from typing import TYPE_CHECKING
import itertools as it

from stream.stages.stage import Stage, StageCallable
from zigzag.utils import open_yaml

if TYPE_CHECKING:
    from stream.cost_model.cost_model import StreamCostModelEvaluation
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)

def calculate_carbon(
    scme: "StreamCostModelEvaluation",
    is_chiplet: bool,
    output_path: str = "outputs/carbon.txt",
):
    #area is in unit: normalized by 1 MAC area for 1 element calculation
    noc_area_list = []
    core_area_list = []
    for core in scme.accelerator.core_list: 
        if core.core_area !=0:
            core_area_list.append(core.core_area)
             
            noc_area_list.append(core.noc_area)
    # if is_chiplet: 
    #     area_list = [x + y for x, y in zip(noc_area_list, core_area_list)]
    # else: 
    #     area_list = core_area_list
    area_list = [x + y for x, y in zip(noc_area_list, core_area_list)]
    lifeop_co2 = scme.carbon
    combinations = list(it.product([scme.carbon_param.technology_node], repeat = len(area_list)))
    area = area_list
    print(area_list)
    design_carbon = np.zeros((len(combinations), len(area_list)+1))
    carbon = np.zeros((len(combinations), len(area_list)+1))

    if is_chiplet:
        for n, comb in enumerate(combinations): 
            cpa = get_carbon_per_area(comb)
            defect_density = get_defect_rate(comb) 
            new_area = area
            total_area = sum(new_area)
            yields = []
            wastage_extra_cfp = [] 
            for i, c in enumerate(comb): 
                yields.append(yield_calc(new_area[i],defect_density[i]))
                wastage_extra_cfp.append(waste_carbon_per_die(diameter=450, chip_area=new_area[i], cpa_factors=cpa[i]))
            # print("chiplet_yield: ", yields)
            # print("chiplet_wastage_extra_cfp", wastage_extra_cfp)
            carbon[n,:-1] = cpa*np.array(new_area) / yields + wastage_extra_cfp   # in g
            design_carbon_per_chiplet, design_carbon[n,:-1] = design_costs(new_area, 8,10,700,comb)
            package_c, router_c, design_carbon[n,-1], router_a = package_costs(scme, area_list, comb, new_area, True, is_chiplet, 700)
            carbon[n, -1] = package_c * 1 + router_c   # in g
    else: 
        cpa = get_carbon_per_area([scme.carbon_param.technology_node])
        defect_density = get_defect_rate([scme.carbon_param.technology_node])
        total_area = sum(area)
        yields = yield_calc(sum(area), defect_density[0])
        # print("yields",yields)
        wastage_extra_cfp = waste_carbon_per_die(diameter=450, chip_area=sum(area), cpa_factors=cpa[0])
        # print("wastage_extra_cfp",wastage_extra_cfp)
        # wastage_extra_cfp =(wastage_extra_cfp * area) / sum(area)
        # carbon[n,:-1] = cpa*sum(area) / yields + wastage_extra_cfp   # in g
        area = np.array(sum(area))

        result = cpa*area/yields + wastage_extra_cfp
    
    # print("----------------carbon--------------------")
    if is_chiplet: 
        print(carbon)
    else:
        print(result)
    # print("----------------ope carbon----------------")
    # print(scme.carbon)
    # print("----------------design carbon--------------")
    # print(design_carbon)
    # print("----------------total area-----------------")
    # print(sum(area))

    

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
    defect_density = defect_density["defect_density"]
    nodes = nodes["technology_node"]
    defect_rate = []
    for c in technology_node: 
        if c in nodes:
            # in node is in 7,10,14,22,28, return defect density directly
            defect_rate.append(defect_density[nodes.index(c)])
        else: 
            defect_rate.append(np.interp(c, nodes, defect_density))
    return defect_rate

def get_carbon_per_area(technology_node):
    cpas = open_yaml("stream/inputs/examples/carbon/carbon_per_area.yaml")
    nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
    cpas = cpas["cpa"]
    nodes = nodes["technology_node"]
    carbon_per_area = []
    for c in technology_node:
        # in node is in 7,10,14,22,28, return cpa directly
        if c in technology_node:
            carbon_per_area.append(cpas[nodes.index(c)])
        # or use interpolation
        else: 
            carbon_per_area.append(np.interp(c, nodes, cpas))
    return carbon_per_area

def area_scaling(area, technology_node): 
    # do area scaling based on digital/sram type
    logic_scaling = open_yaml("stream/inputs/examples/carbon/logic_scaling.yaml")
    sram_scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
    analog_scaling = open_yaml("stream/inputs/examples/carbon/analog_scaling.yaml")
    scalings = list(zip(logic_scaling["area"], analog_scaling["area"], sram_scaling["area"]))
    nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
    nodes = nodes["technology_node"]
    new_area = []
    for index in range(len(technology_node)): 
        if technology_node[index] in nodes: 
            new_area.append(area[index] * scalings[nodes.index(technology_node[index])][index])
        else: 
            new_area.append(area[index]*np.interp(technology_node[index], nodes, scalings[nodes.index(technology_node[index])]))
    return new_area

def design_costs(areas, Transistors_per_gate,Power_per_core,Carbon_per_kWh, technology_node):
    transistor_density = open_yaml("stream/inputs/examples/carbon/transistor_scaling.yaml")
    transistor_density = transistor_density['Transistors_per_mm2']
    gates_design_time = open_yaml("stream/inputs/examples/carbon/gates_per_hr_per_core.yaml")
    gates_design_time = gates_design_time['Gates_per_hr_per_core']
    nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
    nodes = nodes["technology_node"]
    transistors = []
    for index in range(len(technology_node)): 
        if technology_node[index] in nodes: 
            transistors.append(areas[index]*transistor_density[nodes.index(technology_node[index])])
        else: 
            transistors.append(areas[index]*np.interp(technology_node[index], nodes, 
                                    transistor_density[nodes.index(technology_node[index])]))
    gates = [transistor/Transistors_per_gate for transistor in transistors]
    CPU_core_hours = []
    for index in range(len(technology_node)): 
        if technology_node[index] in nodes: 
            CPU_core_hours.append(gates[index]/gates_design_time[nodes.index(technology_node[index])])
        else: 
            CPU_core_hours.append(gates[index]/np.interp(technology_node[index], nodes, 
                                    gates_design_time[nodes.index(technology_node[index])]))
    total_energy = [Power_per_core*CPU_core_hour/1000 for CPU_core_hour in CPU_core_hours] #in kWh
    design_carbon = [Carbon_per_kWh * x for x in total_energy]
    design_carbon_per_chiplet = [x*90/1e5 for x in design_carbon]
    return design_carbon_per_chiplet, design_carbon

def package_costs(scme: "StreamCostModelEvaluation", area_list, technology_node, new_area, return_router_area, is_chiplet, carbon_per_kwh):
    package_carbon = 0 
    router_carbon = 0
    router_design = 0
    router_area = 0
    package_param = open_yaml("stream/inputs/examples/carbon/package_param.yaml")
    bonding_yield = package_param["bonding_yield"]
    new_area = [float(x) for x in new_area]
    if is_chiplet:
        num_chiplets = len(new_area)
        interposer_area, num_if = recursive_split(new_area, emib_pitch=package_param["EMIBPitch"])
        num_if = int(np.ceil(num_if))
        interposer_area = np.prod(interposer_area)
        interposer_carbon = package_mfg_carbon(package_param["interposer_node"], [interposer_area])
        logic_scaling = open_yaml("stream/inputs/examples/carbon/logic_scaling.yaml")
        sram_scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
        analog_scaling = open_yaml("stream/inputs/examples/carbon/analog_scaling.yaml")
        scalings = list(zip(logic_scaling["area"], analog_scaling["area"], sram_scaling["area"]))
        beolVfeol = open_yaml("stream/inputs/examples/carbon/beolVfeol_scaling.yaml")
        beolVfeol = beolVfeol["beolVfeol"]
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        nodes = nodes["technology_node"]
        package = scme.carbon_param.package_type
        if package == "active": 
            router_area = 4.47 * num_chiplets
            router_area = interposer_carbon * router_area/interposer_area
            router_design = 0 
            package_carbon = interposer_carbon * beolVfeol[-1]
        elif package == "3D":
            dims = np.sqrt(np.array(area_list, dtype=np.float64))
            num_tsv_1d = np.floor(dims/package_param["tsv_pitch"])
            overhead_3d = (num_tsv_1d**2) * (package_param["tsv_size"]**2)
            area_3d = area_list + overhead_3d
            carbon3d= package_mfg_carbon(technology_node,area_3d)
            carbon2d= package_mfg_carbon(technology_node,area_list)
            package_carbon = np.sum(carbon3d-carbon2d)
            router_area = []
            for index in range(len(area_list)):
                router_area.append(0.33/scalings[2][index])
            router_carbon= package_mfg_carbon(technology_node, router_area)
            router_carbon= np.sum(router_carbon)
            router_design = 0
            bonding_yield = bonding_yield**num_chiplets
        elif package in ['passive', 'RDL', 'EMIB']: 
            router_area = []
            for index in range(len(area_list)):
                router_area.append(0.33/scalings[2][0])
            # router_carbon= self.package_mfg_carbon(technology_node, router_area)
            cpa = get_carbon_per_area(technology_node) 
            defect_density = get_defect_rate(technology_node)
            # new_area = area_scaling(router_area,technology_node)
            new_area = router_area
            yields = []
            wastage_extra_cfp = []
            if ~(np.all(np.array(technology_node) == technology_node[0])): 
                for i, c in enumerate(technology_node):   
                    yields.append(yield_calc(new_area[i], defect_density[i]))
                    wastage_extra_cfp.append(waste_carbon_per_die(diameter=450,chip_area=new_area[i],cpa_factors=cpa[i]))
            else: 
                yields = yield_calc(sum(new_area), defect_density[0])
                wastage_extra_cfp = waste_carbon_per_die(diameter=450,chip_area=sum(new_area),cpa_factors=cpa[0])
                wastage_extra_cfp = np.array(wastage_extra_cfp)
                wastage_extra_cfp = (wastage_extra_cfp * router_area) / sum(router_area)
            router_carbon = cpa*np.array(new_area) / yields# + wastage_extra_cfp
            design_carbon_per_chiplet, router_design = design_costs(new_area, 8, 10, 700, technology_node)
            router_carbon, router_design = np.sum(router_carbon), np.sum(router_design)
            if package == 'passive':
                package_carbon = interposer_carbon* beolVfeol[nodes.index(package_param["interposer_node"])]
            elif package == 'RDL':
                package_carbon = interposer_carbon* beolVfeol[nodes.index(package_param["interposer_node"])]
                package_carbon *= package_param["RDLLayers"]/package_param["numBEOL"]   
            elif package == 'EMIB':
                emib_area =  [5*5]*num_if
                emib_carbon = package_mfg_carbon([22]*num_if,emib_area)
                package_carbon = np.sum(emib_carbon)* beolVfeol[3] #22nm
        else: 
            raise NotImplemented      
    package_carbon /= bonding_yield
    router_carbon /= bonding_yield
    if return_router_area: 
        return package_carbon, router_carbon, router_design, router_area
    else: 
        return package_carbon, router_carbon, router_design
    
def recursive_split(areas, axis=0, emib_pitch=10):
    sorted_areas = np.sort(areas[::-1])
    if len(areas)<=1:
        v = (np.sum(areas)/2)**0.5
        size_2_1 = np.array((v + v*((axis+1)%2), v +axis*v))
        return size_2_1, 0
    else:
        sums = np.array((0.0,0.0))
        blocks= [[],[]]
        for i, area in enumerate(sorted_areas):
            blocks[np.argmin(sums)].append(area)
            sums[np.argmin(sums)] += area
        left, l_if = recursive_split(blocks[0], (axis+1)%2, emib_pitch)
        right, r_if = recursive_split(blocks[1], (axis+1)%2, emib_pitch)
        sizes = np.array((0.0,0.0))
        sizes[axis] = left[axis] + right[axis] + 0.5
        sizes[(axis+1)%2] = np.max((left[(axis+1)%2], right[(axis+1)%2]))
        t_if = l_if + r_if 
        t_if += np.ceil(np.min((left[(axis+1)%2], right[(axis+1)%2]))/emib_pitch) # for overlap 1 interface per 10mm
        return sizes, t_if
    
def package_mfg_carbon(technology_node, interposer_area): 
    # print(technology_node, interposer_area)
    cpa = get_carbon_per_area([technology_node]) 
    defect_density = get_defect_rate([technology_node])
    # print("defect_den:, ", defect_density[0]/4)
    new_area = interposer_area
    # new_area = self.area_scaling(interposer_area, [technology_node])
    total_area = sum(new_area)
    yields = []
    wastage_extra_cfp = []
    yields = yield_calc(sum(new_area), defect_density[0]/4)
    """
    wastage_extra_cfp = self.waste_carbon_per_die(diameter=450,chip_area=sum(new_area),cpa_factors=cpa[0])
    wastage_extra_cfp = (wastage_extra_cfp * new_area[0]) / sum(new_area)
    """
    interposer_carbon = cpa*np.array(new_area) / yields #  + wastage_extra_cfp
    return interposer_carbon





    
    
    
    


     