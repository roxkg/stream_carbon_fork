import numpy as np
import pandas as pd
import itertools as it
import math
import ast
from typing import Any
from matplotlib import pyplot as plt
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
        # op_co2 = op_CI * energy_value / (3.6 * 10**18)
        lifespan = self.scme.lifetime
        frequency = self.scme.frequency
        # taskspan = self.scme.latency/(frequency*10**9)   
        # year -> s 
        # lifetime working proportion: 0.2
        # lifeop_co2 = lifespan*365*24*60*60*0.2/taskspan * op_co2
        lifeop_co2 = lifespan*365*24*60*60*0.2 * op_CI
        combinations = list(it.product(self.scme.technology_node, repeat=len(self.scme.area_list)))
        area = np.array(list(self.scme.area_list.values()))
        design_carbon = np.zeros((len(combinations), len(self.scme.area_list)+1))
        op_carbon = np.zeros((len(combinations), len(self.scme.area_list)+1))
        carbon = np.zeros((len(combinations), len(self.scme.area_list)+1))
        emb_carbon = np.zeros(len(combinations))
        op_carbon_1 = np.zeros(len(combinations))
        for n, comb in enumerate(combinations):
            cpa = self.get_carbon_per_area(comb) 
            defect_density = self.get_defect_rate(comb)
            new_area = self.area_scaling(area, comb)
            total_area = sum(new_area)
            yields = []
            wastage_extra_cfp = []
            if ~(np.all(np.array(comb) == comb[0])): 
                for i, c in enumerate(comb):   
                    yields.append(self.yield_calc(new_area[i], defect_density[i]))
                    wastage_extra_cfp.append(self.waste_carbon_per_die(diameter=450,chip_area=new_area[i],cpa_factors=cpa[i]))
            else: 
                yields = self.yield_calc(sum(new_area), defect_density[0])
                wastage_extra_cfp = self.waste_carbon_per_die(diameter=450,chip_area=sum(new_area),cpa_factors=cpa[0])
                wastage_extra_cfp = (wastage_extra_cfp * area) / area.sum()

            carbon[n,:-1] = cpa*np.array(new_area) / yields + wastage_extra_cfp
            # mfg_carbon = cpa*np.array(new_area) / yields
            # design_carbon_per_chiplet, design_carbon = self.design_costs(new_area,8,10,700,comb)
            design_carbon_per_chiplet, design_carbon[n,:-1] = self.design_costs(new_area,8,10,700,comb)
            design_carbon[n,:-1] = design_carbon[n,:-1]*90/1e5
            package_c, router_c, design_carbon[n,-1] = self.package_costs(comb, False, 700)
            carbon[n, -1] = package_c*1 + router_c   # 1 is package factor
            activity=[0.2, 0.667, 0.1]
            op_carbon[n,:-1] = self.operational_costs(activity, comb, area*self.scme.energy_use/sum(area), self.scme.lifetime, 700)
            emb_carbon[n:-1] = sum(carbon[n,:-1]) + sum(design_carbon[n,:-1])
            op_carbon_1[n:-1] = sum(op_carbon[n,:-1])
        """
        total_carbon = carbon + design_carbon + op_carbon
        carbon = pd.DataFrame(data=carbon, index=combinations, columns=(list(self.scme.area_list.keys()) + ["Packaging"]))
        total_carbon = pd.DataFrame(data=total_carbon, index=combinations, columns=(list(self.scme.area_list.keys()) + ["Packaging"]))
        carbon.plot(kind='bar', stacked=True, figsize = (21,7),
            title=f'Stacked CO2 manufacturing: GA102')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        total_carbon = pd.DataFrame(data=zip(op_carbon_1,emb_carbon), index=combinations, columns=(["op","emb"]))
        total_carbon.plot(kind='bar', stacked=True, figsize = (10,7),
            title=f'Total C02 manufacturing+design: GA102')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()
        """

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
    
    def package_costs(self, technology_node, return_router_area, carbon_per_kwh):
        package_carbon = 0 
        router_carbon = 0
        router_design = 0
        router_area = 0
        package_param = open_yaml("stream/inputs/examples/carbon/package_param.yaml")
        bonding_yield = package_param["bonding_yield"]
        if ~(np.all(np.array(technology_node) == technology_node[0])):
            num_chiplets = len(self.scme.area_list)
            print(list(self.scme.area_list.values()))
            interposer_area, num_if = self.recursive_split(list(self.scme.area_list.values()), emib_pitch=package_param["EMIBPitch"])
            num_if = int(np.ceil(num_if))
            interposer_area = np.prod(interposer_area)
            print(interposer_area)
            interposer_carbon = self.package_mfg_carbon(package_param["interposer_node"], [interposer_area])

            logic_scaling = open_yaml("stream/inputs/examples/carbon/logic_scaling.yaml")
            sram_scaling = open_yaml("stream/inputs/examples/carbon/sram_scaling.yaml")
            analog_scaling = open_yaml("stream/inputs/examples/carbon/analog_scaling.yaml")
            scalings = list(zip(logic_scaling["area"], analog_scaling["area"], sram_scaling["area"]))
            beolVfeol = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
            beolVfeol = beolVfeol["beolVfeol"]
            nodes = open_yaml("stream/inputs/examples/carbon/technology_node.yaml")
            nodes = nodes["technology_node"]
            if self.scme.package_type=="active": 
                router_area = 4.47 * num_chiplets
                router_area = interposer_carbon * router_area/interposer_area
                router_design = 0 
                package_carbon = interposer_carbon * beolVfeol[-1]

            elif self.scme.package_type == "3D":
                dims = np.sqrt(np.array(self.scme.area_list, dtype=np.float64))
                num_tsv_1d = np.floor(dims/package_param["tsv_pitch"])
                overhead_3d = (num_tsv_1d**2) * (package_param["tsv_size"]**2)
                area_3d = self.scme.area_list + overhead_3d
                carbon3d= self.package_mfg_carbon(technology_node,area_3d)
                carbon2d= self.package_mfg_carbon(technology_node,self.scme.area_list)
                package_carbon = np.sum(carbon3d-carbon2d)
                
                router_area = 0.33/np.array([scalings[ty][14, 'area'] for ty in self.scme.area_list.keys()])
                router_carbon, router_design, _ = self.package_mfg_carbon(technology_node, router_area)
                router_carbon, router_design = np.sum(router_carbon), np.sum(router_design) 
                bonding_yield = bonding_yield**num_chiplets
            elif self.scme.package_type in ['passive', 'RDL', 'EMIB']: 
                router_area = 0.33/np.array([scalings[ty].loc[14, 'area'] for ty in self.scme.area_list.keys()])
                router_carbon= self.package_mfg_carbon(technology_node, router_area)
                router_design = 0
                router_carbon, router_design = np.sum(router_carbon), np.sum(router_design)
                if self.scme.package_type == 'passive':
                    package_carbon = interposer_carbon* beolVfeol[nodes.index(package_param["interposer_node"])]
                elif self.scme.package_type == 'RDL':
                    package_carbon = interposer_carbon* beolVfeol[nodes.index(package_param["interposer_node"])]
                    package_carbon *= package_param["RDLLayers"]/package_param["numBEOL"]   
                elif self.scme.package_type == 'EMIB':
                    emib_area =  [5*5]*num_if
#                   print("NUMBER OF INTERFACES",num_if)
                    emib_carbon = self.package_mfg_carbon([22]*num_if,emib_area)
                    package_carbon = np.sum(emib_carbon)* beolVfeol[3] #22nm
            else: 
                raise NotImplemented
            
        package_carbon /= bonding_yield
        router_carbon /= bonding_yield
        if return_router_area: 
            return package_carbon, router_carbon, router_design, router_area
        else: 
            return package_carbon, router_carbon, router_design


    def recursive_split(self, areas, axis=0, emib_pitch=10):
        sorted_areas = np.sort(areas[::-1])
        if len(areas)<=1:
            v = (np.sum(areas)/2)**0.5
            size_2_1 = np.array((v + v*((axis+1)%2), v +axis*v))
#             print("single", axis, size_2_1)
            return size_2_1, 0
        else:
            sums = np.array((0.0,0.0))
            blocks= [[],[]]
            for i, area in enumerate(sorted_areas):
                blocks[np.argmin(sums)].append(area)
                sums[np.argmin(sums)] += area
#           print("blocks",axis, blocks)
            left, l_if = self.recursive_split(blocks[0], (axis+1)%2, emib_pitch)
#           print("left",axis, left)
            right, r_if = self.recursive_split(blocks[1], (axis+1)%2, emib_pitch)
#           print("right",axis, right)
            sizes = np.array((0.0,0.0))
            sizes[axis] = left[axis] + right[axis] + 0.5
            sizes[(axis+1)%2] = np.max((left[(axis+1)%2], right[(axis+1)%2]))
            t_if = l_if + r_if 
            t_if += np.ceil(np.min((left[(axis+1)%2], right[(axis+1)%2]))/emib_pitch) # for overlap 1 interface per 10mm
            return sizes, t_if
    
    def package_mfg_carbon(self, technology_node, interposer_area): 
        cpa = self.get_carbon_per_area([technology_node]) 
        defect_density = self.get_defect_rate([technology_node])
        new_area = interposer_area
        total_area = sum(new_area)
        yields = []
        wastage_extra_cfp = []
        if ~(np.all(np.array(technology_node) == [technology_node][0])): 
            for i, c in enumerate(technology_node):   
                yields.append(self.yield_calc(new_area[i], defect_density[i]/4))
                wastage_extra_cfp.append(self.waste_carbon_per_die(diameter=450,chip_area=new_area[i],cpa_factors=cpa[i]))
        else: 
            yields = self.yield_calc(sum(new_area), defect_density[0]/4)
            wastage_extra_cfp = self.waste_carbon_per_die(diameter=450,chip_area=sum(new_area),cpa_factors=cpa[0])
            wastage_extra_cfp = (wastage_extra_cfp * interposer_area) / interposer_area.sum()

        interposer_carbon = cpa*np.array(new_area) / yields + wastage_extra_cfp
        return interposer_carbon
    

input_data = open_yaml("stream/inputs/testing/carbon_validation/GA102.yaml")
area_dict = {item["type"]: item["area"] for item in input_data["area_list"]}

hardware = CarbonModel(CI_op=input_data["CI_op"], 
                       CI_em=input_data["CI_em"], 
                       lifetime=input_data["lifetime"], 
                       frequency=input_data["frequency"], 
                       technology_node=input_data["technology_node"], 
                       area_list=area_dict,
                       energy_use=input_data["energy_op"], 
                       scaling_enable=True, 
                       package_type=input_data["package_type"])

evaluation= CarbonEvaluation(hardware)
evaluation.calculate_carbon("outputs/test_result/TPU.txt")



    








    
    
    
    


     