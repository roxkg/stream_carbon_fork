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
            mem_area_origin = self.scme.area_list.get("mem_area")
            # total_area = self.scme.area_total
            logic_area_origin = self.scme.area_list.get("logic_area")
            analog_area_origin = self.scme.area_list.get("analog_area")
            area_origin = mem_area_origin + logic_area_origin + analog_area_origin
            logic_area,mem_area,analog_area = self.area_scaling(logic_area_origin, mem_area_origin, analog_area_origin, node['node'])
            area = logic_area + mem_area + analog_area
    
            wastage_extra_cfp = self.waste_carbon_per_die(diameter=450,chip_area=area,cpa_factors=cpa)
            yields = self.yield_calc(area,defect_density)
            mfg_carbon = cpa*area / yields
            # wastage_extra_cfp_logic = (wastage_extra_cfp * logic_area) / area
            # wastage_extra_cfp_mem = (wastage_extra_cfp * mem_area) / area
            mfg_carbon = mfg_carbon+wastage_extra_cfp
            # mfg_carbon = mfg_carbon+wastage_extra_cfp_logic+ wastage_extra_cfp_mem
            print("-------------mfg_carbon----------------")
            print(node,"nm:",mfg_carbon,"mfg_carbon(g)")

            design_carbon_per_chiplet, total_design_carbon = self.design_costs(area,8, 10, 700, node['node']  )
            logic_carbon_per_chiplet, total_logic_carbon = self.design_costs(logic_area,8, 10, 700, node['node'] )
            mem_carbon_per_chiplet, total_mem_carbon = self.design_costs(mem_area,8, 10, 700, node['node'] )
            analog_carbon_per_chiplet, total_analog_carbon = self.design_costs(analog_area, 8, 10, 700, node['node'])
            print("------------design carbon---------------")
            print(node,"nm:",design_carbon_per_chiplet,"design_carbon_per_chiplet(g)" )
            print(node,"nm:",total_design_carbon,"total_design_carbon(g)" )
            print(node,"nm:",logic_carbon_per_chiplet,"design_carbon_per_logic_chiplet(g)" )
            print(node,"nm:",total_logic_carbon,"total_logic_design_carbon(g)" )
            print(node,"nm:",mem_carbon_per_chiplet,"design_carbon_per_mem_chiplet(g)" )
            print(node,"nm:",total_mem_carbon,"total_mem_design_carbon(g)" )
            print(node,"nm:",analog_carbon_per_chiplet,"analog_carbon_per_chiplet(g)" )
            print(node,"nm:",total_analog_carbon,"total_analog_carbon(g)" )
            print("--------op carbon---------")

            activity=[0.2, 0.667, 0.1]
            op_carbon_logic = self.operational_costs(activity, node['node'], "logic", logic_area_origin*self.scme.energy_use/area_origin, self.scme.lifetime, 700)
            print(node,"nm:",op_carbon_logic,"op_carbon_logic(kg)" )
            op_carbon_sram = self.operational_costs(activity, node['node'], "sram", mem_area_origin*self.scme.energy_use/area_origin, self.scme.lifetime, 700)
            print(node,"nm:",op_carbon_sram,"op_carbon_sram(kg)" )
            op_carbon = op_carbon_logic + op_carbon_sram
            print(node,"nm:",op_carbon,"op_carbon(kg)" )

            print("------------------total-----------------")
            total = (mfg_carbon + design_carbon_per_chiplet)/1000
            print("total co2 kg: ", total)
            
            print("--------------------------------------------")

        

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
    
    def area_scaling(self,logic_area, mem_area, analog_area, technology_node): 
        # do area scaling based on digital/sram type
        logic_scaling = open_yaml("stream/inputs/examples/carbon/logic_scaling.yaml")
        sram_scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
        analog_scaling = open_yaml("stream/inputs/examples/carbon/analog_scaling.yaml")
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        nodes = nodes["technology_node"]
        if technology_node in nodes: 
            logic_area = logic_area * logic_scaling["area"][nodes.index(technology_node)]
            mem_area = mem_area * sram_scaling["area"][nodes.index(technology_node)]
            analog_area = analog_area * analog_scaling["area"][nodes.index(technology_node)]
            return logic_area,mem_area, analog_area

        logic_area_scaing = np.interp(technology_node, nodes, logic_scaling["area"])
        sram_area_scaing = np.interp(technology_node, nodes, sram_scaling["area"])
        analog_area_scaing = np.interp(technology_node, nodes, analog_scaling["area"])
        logic_area = logic_area * logic_area_scaing
        mem_area = mem_area * sram_area_scaing
        analog_area = analog_area * analog_area_scaing
        return logic_area,mem_area, analog_area
    
    def design_costs(self, areas, Transistors_per_gate,Power_per_core,Carbon_per_kWh, technology_node):
        transistor_density = open_yaml("stream/inputs/examples/carbon/transistor_scaling.yaml")
        gates_design_time = open_yaml("stream/inputs/examples/carbon/gates_per_hr_per_core.yaml")
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        nodes = nodes["technology_node"]
        transistors = areas * transistor_density['Transistors_per_mm2'][nodes.index(technology_node)]
        gates = transistors/Transistors_per_gate
        CPU_core_hours = gates / gates_design_time['Gates_per_hr_per_core'][nodes.index(technology_node)]
        total_energy = Power_per_core*CPU_core_hours/1000 #in kWh
        design_carbon = Carbon_per_kWh * total_energy
        design_carbon_per_chiplet = design_carbon*90/1e5
        return design_carbon_per_chiplet, design_carbon
    
    def operational_costs(self, activity, technology_node, type, powers_in, lifetime, op_CI):
        active = activity[0]
        on = activity[1]
        avg_pwr = activity[2]
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        nodes = nodes["technology_node"]
        dyn_pwr_ratio = open_yaml("stream/inputs/examples/carbon/dyn_pwr_ratio.yaml")
        if type == "logic": 
            scaling = open_yaml("stream/inputs/examples/carbon/logic_scaling.yaml")
        else:
            scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
        dyn_ratio =  dyn_pwr_ratio['dyn_pwr_ratio'][nodes.index(technology_node)]
        pwr_scale = scaling["power"][nodes.index(technology_node)]
        powers_tech_scaled = powers_in * pwr_scale
        powers_scaled = powers_tech_scaled*on*avg_pwr*(dyn_ratio*active + (1-dyn_ratio))
        energy = lifetime*powers_scaled/1000
        op_carbon = op_CI * energy
        return op_carbon
    
input_data = open_yaml("stream/inputs/testing/carbon_validation/GA102.yaml")
area_dict = {item["name"]: item["area"] for item in input_data["area_list"]}

hardware = CarbonModel(CI_op=input_data["CI_op"], 
                       CI_em=input_data["CI_em"], 
                       lifetime=input_data["lifetime"], 
                       frequency=input_data["frequency"], 
                       technology_node=input_data["technology_node"], 
                       area_list=area_dict,
                       energy_use=input_data["energy_op"])

evaluation= CarbonEvaluation(hardware)
evaluation.calculate_carbon("outputs/test_result/GA102.txt")



    








    
    
    
    


     