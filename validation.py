import numpy as np
import pandas as pd
import itertools as it
import math
import ast
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

        combinations = list(it.product(self.scme.technology_node, repeat=len(self.scme.technology_node)))
        area = np.array(list(self.scme.area_list.values()))
        for n, comb in enumerate(combinations):
            cpa = self.get_carbon_per_area(comb) 
            defect_density = self.get_defect_rate(comb)
            new_area = self.area_scaling(area, comb)
            total_area = sum(new_area)
            yields = []
            wastage_extra_cfp = []
            for i, c in enumerate(comb):   
                yields.append(self.yield_calc(new_area[i], defect_density[i]))
                wastage_extra_cfp.append(self.waste_carbon_per_die(diameter=450,chip_area=new_area[i],cpa_factors=cpa[i]))
            mfg_carbon = cpa*area / yields
            # print(mfg_carbon + wastage_extra_cfp)

            design_carbon_per_chiplet, design_carbon = self.design_costs(new_area,8,10,700,comb)
            # print(design_carbon_per_chiplet)
            # print(design_carbon)

            activity=[0.2, 0.667, 0.1]
            op_carbon = self.operational_costs(activity, comb, area*self.scme.energy_use/sum(area), self.scme.lifetime, 700)
            print(op_carbon)

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
        defect_rate = []
        for c in technology_node: 
            if c in nodes:
                defect_rate.append(defect_density[nodes.index(c)])
            else: 
                defect_rate.append(np.interp(c, nodes, defect_density))

        return defect_rate

    def get_carbon_per_area(self,technology_node):
        cpas = open_yaml("stream/inputs/examples/carbon/carbon_per_area.yaml")
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        cpas = cpas["cpa"]
        
        nodes = nodes["technology_node"]
        carbon_per_area = []
        # carbon_per_area = np.zeros(len(technology_node))
        for c in technology_node: 
            if c in nodes:
                carbon_per_area.append(cpas[nodes.index(c)])
            else: 
                carbon_per_area.append(np.interp(c, nodes, cpas))
        # in node is in 7,10,14,22,28, return cpa directly
        # or use interpolation
        return carbon_per_area
    
    def area_scaling(self,area, technology_node):  
        # do area scaling based on digital/sram/analog type
        logic_scaling = open_yaml("stream/inputs/examples/carbon/logic_scaling.yaml")
        sram_scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
        analog_scaling = open_yaml("stream/inputs/examples/carbon/analog_scaling.yaml")
        scalings = list(zip(logic_scaling["area"], analog_scaling["area"], sram_scaling["area"]))
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        nodes = nodes["technology_node"]
        new_area = []
        for index in range(len(technology_node)):
            if technology_node[index] in nodes: 
                new_area.append(area[index]* 
                                scalings[nodes.index(technology_node[index])][index])
            else: 
                new_area.append(area[index]* 
                                np.interp(technology_node[index], nodes, scalings[nodes.index(technology_node[index])]))
        return new_area    

    
    def design_costs(self, areas, Transistors_per_gate,Power_per_core,Carbon_per_kWh, technology_node):
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
    
    def operational_costs(self, activity, technology_node, powers_in, lifetime, op_CI):
        active = activity[0]
        on = activity[1]
        avg_pwr = activity[2]
        nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
        nodes = nodes["technology_node"]
        dyn_pwr_ratio = open_yaml("stream/inputs/examples/carbon/dyn_pwr_ratio.yaml")
        dyn_pwr_ratio = dyn_pwr_ratio['dyn_pwr_ratio']
        logic_scaling = open_yaml("stream/inputs/examples/carbon/logic_scaling.yaml")
        sram_scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
        analog_scaling = open_yaml("stream/inputs/examples/carbon/analog_scaling.yaml")
        scalings = list(zip(logic_scaling["power"], analog_scaling["power"], sram_scaling["power"]))
        dyn_ratio = []
        pwr_scale = []
        for index in range(len(technology_node)): 
            dyn_ratio.append(dyn_pwr_ratio[nodes.index(technology_node[index])])
            pwr_scale.append(scalings[nodes.index(technology_node[index])][index])
        
        powers_tech_scaled = powers_in * pwr_scale
        powers_scaled = powers_tech_scaled*on*avg_pwr*(np.array(dyn_ratio)*active + (1-np.array(dyn_ratio)))
        energy = lifetime*powers_scaled/1000
        op_carbon = op_CI * energy
        return op_carbon
    
input_data = open_yaml("stream/inputs/testing/carbon_validation/TPU.yaml")
area_dict = {item["type"]: item["area"] for item in input_data["area_list"]}

hardware = CarbonModel(CI_op=input_data["CI_op"], 
                       CI_em=input_data["CI_em"], 
                       lifetime=input_data["lifetime"], 
                       frequency=input_data["frequency"], 
                       technology_node=input_data["technology_node"], 
                       area_list=area_dict,
                       energy_use=input_data["energy_op"], 
                       scaling_enable=True)

evaluation= CarbonEvaluation(hardware)
evaluation.calculate_carbon("outputs/test_result/GA102.txt")



    








    
    
    
    


     