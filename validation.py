import numpy as np
import pandas as pd
import argparse
import logging
import yaml
import pickle
import math
from typing import Any
from zigzag.utils import open_yaml
from validation_carbon_model import CarbonModel

class CarbonEvaluation: 
    def __init__(self, scme: CarbonModel):
        self.scme = scme

    def calculate_carbon(
        self,
        output_path: str = "outputs/carbon.txt",
    ):
        #area is in unit: normalized by 1 MAC area for 1 element calculation
        # total_area = self.scme.area_total
        # mem_area = self.scme.area_list["mem_area"]
        op_CI = self.scme.CI_op
        # energy_value = float(self.scme.energy)
        # pJ -> kWh
        # op_co2 = op_CI * energy_value / (3.6 * 10**18)
        lifespan = self.scme.lifetime
        frequency = self.scme.frequency
        # taskspan = self.scme.latency/(frequency*10**9)   
        # year -> s 
        # lifetime working proportion: 0.2
        # lifeop_co2 = lifespan*365*24*60*60*0.2/taskspan * op_co2
        lifeop_co2 = lifespan*365*24*60*60*0.2 * op_CI
        # current technology node: 16nm -> approximated by 14nm 
        # all data from ECO-CHIP
        #defect_density = 0.09
        for node in self.scme.technology_node:
            #defect_density = self.get_defect_rate(self.scme.carbon_param.technology_node)
            defect_density = self.get_defect_rate(node['node'])
            # one large chip
            cpa = self.get_carbon_per_area(node['node']) 
            # area = (12000/1024*total_area)*0.000008*0.000008
            # print(area)
            mem_area = self.scme.area_list.get("mem_area")
            # total_area = self.scme.area_total
            logic_area = self.scme.area_list.get("logic_area")
            logic_area,mem_area = self.area_scaling(logic_area, mem_area, node['node'])
            area = logic_area + mem_area
    
            wastage_extra_cfp = self.waste_carbon_per_die(diameter=450,chip_area=area,cpa_factors=cpa)
            yields = self.yield_calc(area,defect_density)
            mfg_carbon = cpa*area / yields
            mfg_carbon = mfg_carbon+wastage_extra_cfp
            print(mfg_carbon)

        

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
            # file.write(f"op_co2 = {op_co2}g\ntotal life co2 = {lifeop_co2}\n area={total_area}\n")
            file.write(f"em_co2 = {cemb}")
        # RDL fan-out 
        # "rdl_layers" = 6, "defect_density" = 0.09 

    def yield_calc(self,area, defect_density):
        yield_val = (1+(defect_density*1e4)*(area*1e-6)/10)**-10
        return yield_val

    def waste_carbon_per_die(self,diameter,chip_area,cpa_factors):
        wafer_area = (math.pi * (diameter ** 2))/4
        dies_per_wafer = math.pi * diameter * ((diameter/(4*chip_area)) - (1/math.sqrt(2*chip_area)))
        area_wastage = wafer_area - (math.floor(dies_per_wafer)*chip_area)
        #unused_si_cfp = wastage_cfp(area_wastage,techs,scaling_factors)
        unused_si_cfp = area_wastage*cpa_factors
        wastage_si_cfp_pdie = unused_si_cfp/dies_per_wafer
        return wastage_si_cfp_pdie

    def get_defect_rate(self,technology_node):
        defect_density = open_yaml("stream/inputs/examples/carbon/defect_density.yaml")
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        # in node is in 7,10,14,22,28, return defect density directly
        defect_density = defect_density["defect_density"]
        nodes = nodes["technology_node"]
        if technology_node in nodes:
            return defect_density[nodes.index(technology_node)]

        # or use interpolation
        defect_rate = np.interp(technology_node, nodes, defect_density)

        return defect_rate

    def get_carbon_per_area(self,technology_node):
        cpas = open_yaml("stream/inputs/examples/carbon/carbon_per_area.yaml")
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        cpas = cpas["cpa"]
        
        nodes = nodes["technology_node"]
        # in node is in 7,10,14,22,28, return cpa directly
        if technology_node in nodes:
            return cpas[nodes.index(technology_node)]

        # or use interpolation
        carbon_per_area = np.interp(technology_node, nodes, cpas)

        return carbon_per_area
    def area_scaling(self,logic_area, mem_area, technology_node): 
        # do area scaling based on digital/sram type
        logic_scaling = open_yaml("stream/inputs/examples/carbon/logic_scaling.yaml")
        sram_scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        nodes = nodes["technology_node"]
        if technology_node in nodes: 
            logic_area = logic_area * logic_scaling["area"][nodes.index(technology_node)]
            mem_area = mem_area * sram_scaling["area"][nodes.index(technology_node)]
            return logic_area,mem_area

        logic_area = logic_area * logic_scaling["area"][nodes.index(technology_node)]
        mem_area = mem_area * sram_scaling["area"][nodes.index(technology_node)]
        return logic_area,mem_area
    
input_data = open_yaml("stream/inputs/testing/carbon_validation/GA102.yaml")
area_dict = {item["name"]: item["area"] for item in input_data["area_list"]}

hardware = CarbonModel(CI_op=input_data["CI_op"], 
                       CI_em=input_data["CI_em"], 
                       lifetime=input_data["lifetime"], 
                       frequency=input_data["frequency"], 
                       technology_node=input_data["technology_node"], 
                       area_list=area_dict)

evaluation= CarbonEvaluation(hardware)
evaluation.calculate_carbon("outputs/test_result/GA102.txt")



    








    
    
    
    


     