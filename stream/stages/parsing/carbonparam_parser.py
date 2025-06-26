import logging
import math
import itertools as it
import numpy as np
from typing import Any

from zigzag.utils import open_yaml

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.carbon_validator import CarbonValidator
from stream.parser.carbon_factory import CarbonFactory
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)

class CarbonParamParserStage(Stage): 
    """Parse to parse carbon parameter from a user-defined yaml file."""
    
    def __init__(self, 
        list_of_callables: list[StageCallable], 
        *, 
        carbon_path: str,
        accelerator: Accelerator,
        is_chiplet: bool,
        interposer_area:float,
        rcy_mat_frac:float,
        rcy_cpa_frac:float,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.is_chiplet = is_chiplet
        self.interposer_area = interposer_area
        self.rcy_mat_frac = rcy_mat_frac
        self.rcy_cpa_frac = rcy_cpa_frac
        assert carbon_path.split(".")[-1] == "yaml", "Expected a yaml file as carbon parameter input"
        # build CarbonParam based on input yaml
        carbon_data = open_yaml(carbon_path)
        validator = CarbonValidator(carbon_data, carbon_path)
        self.carbonparam_data = validator.normalized_data  # store data after validation
        validate_success = validator.validate()
        if not validate_success:
            raise ValueError("Failed to validate user provided accelerator.")
        factory = CarbonFactory(self.carbonparam_data)
        self.carbonparam = factory.create()  # create CarbonParam instance
        self.data = self.get_scaling_parameter()

    def run(self): 
        embodied_carbon = self.embodied_carbon_costs(self.is_chiplet)
        self.carbonparam.set_cemb(embodied_carbon)
        self.kwargs["carbon_param"] = self.carbonparam
        self.kwargs["embodied_carbon"] = embodied_carbon[0]
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=self.accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def get_scaling_parameter(self):
        data = {}
        files = ["defect_density","technology_node", "carbon_per_area", "transistor_scaling"
                 , "gates_per_hr_per_core" , "package_param", "beolVfeol_scaling", "logic_scaling", 
                 "sram_scaling", "analog_scaling"]
        for file in files:
            result = open_yaml("stream/inputs/examples/carbon/"+file+".yaml")
            data[file] = result
        return data
    
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
    
    def get_carbon_per_area(self,technology_node):
        cpas = self.data["carbon_per_area"]["cpa"]
        nodes = self.data["technology_node"]["technology_node"]
        carbon_per_area = []
        for c in technology_node:
            # in node is in 7,10,14,22,28, return cpa directly
            if c in nodes:
                carbon_per_area.append(cpas[nodes.index(c)])
            # or use interpolation
            else: 
                carbon_per_area.append(np.interp(c, nodes, cpas))
        return carbon_per_area
    
    def get_defect_rate(self,technology_node):
        defect_density = self.data["defect_density"]["defect_density"]
        nodes = self.data["technology_node"]["technology_node"]
        defect_rate = []
        for c in technology_node: 
            if c in nodes:
                # in node is in 7,10,14,22,28, return defect density directly
                defect_rate.append(defect_density[nodes.index(c)])
            else: 
                defect_rate.append(np.interp(c, nodes, defect_density))
        return defect_rate
    
    def embodied_carbon_costs(self, is_chiplet):
        noc_area_list = []
        core_area_list = []
        core_id_list = []
        for core in self.accelerator.core_list: 
            if core.core_area != 0:
                core_area_list.append(core.core_area)
                noc_area_list.append(core.noc_area)
                core_id_list.append(core.id)
        if is_chiplet: 
            area_list = [ x + y for x, y in zip(noc_area_list, core_area_list)]
            # 12core_1area
            # corearea = core_area_list[0]
            # refactor = corearea*2/4
            # totalarea = refactor + 84*0.86
            # area_list = [totalarea, totalarea, totalarea, totalarea
                        #  ,totalarea, totalarea, totalarea, totalarea,
                        #  totalarea, 
                        #  totalarea, totalarea, totalarea,totalarea, totalarea, totalarea, totalarea,
                        #  totalarea, totalarea, totalarea, totalarea,totalarea, totalarea, totalarea, totalarea,
                        #  totalarea, totalarea, totalarea, totalarea,totalarea, totalarea, totalarea, totalarea,
                        #  totalarea, totalarea, totalarea, totalarea,
                        #  ]
            # noc_area_list = [84*0.86,84*0.86,84*0.86,84*0.86,
                            #  84*0.86,84*0.86,84*0.86,84*0.86,
                            #  84*0.86,
                            #  84*0.45,84*0.45,84*0.45,84*0.45,84*0.45,84*0.45,84*0.45,
                            #  84*0.16,84*0.16,84*0.16,84*0.16,84*0.16,84*0.16,84*0.16,84*0.16,
                            #  84*0.16,84*0.16,84*0.16,84*0.16,84*0.16,84*0.16,84*0.16,84*0.16,
                            #  84*0.16,84*0.16,84*0.16,84*0.16,
                            #  ]
            # core_area_list = [refactor,refactor,refactor,refactor,
                            #   refactor,refactor,refactor,refactor,
                            #   refactor,
                            #   refactor,refactor,refactor,refactor,refactor,refactor,refactor,
                            #   refactor,refactor,refactor,refactor,refactor,refactor,refactor,refactor,
                            #   refactor,refactor,refactor,refactor,refactor,refactor,refactor,refactor,
                            #   refactor,refactor,refactor,refactor,
                            #   ]

            corearea = core_area_list[0]
            refactor = corearea*36/16
            totalarea = refactor + 2.08
            area_list = [totalarea, totalarea, totalarea, totalarea,
                         totalarea, totalarea, totalarea, totalarea,
                         totalarea, totalarea, totalarea, totalarea,
                         totalarea, totalarea, totalarea, totalarea,]
            noc_area_list = [2.08, 2.08, 2.08, 2.08, 
                             2.08, 2.08, 2.08, 2.08,
                             2.08, 2.08, 2.08, 2.08,
                             2.08, 2.08, 2.08, 2.08,]
            core_area_list = [refactor,refactor, refactor, refactor,
                              refactor, refactor, refactor, refactor,
                              refactor, refactor, refactor, refactor,
                              refactor, refactor, refactor, refactor,]
            
            # breakpoint()
            # # 9 core_1area
            # area_list = [209.78,209.78,209.78,209.78,209.78,209.78,209.78,209.78,209.78]
            # noc_area_list = [84,84,84,84,84,84,84,84,84]
            # core_area_list = [125.78,125.78,125.78,125.78,125.78,125.78,125.78,125.78,125.78]
        
        else: 
            area_list = core_area_list
        combinations = list(it.product([self.carbonparam.technology_node], repeat = len(area_list)))
        print("area list: ", area_list, "combinations: ", combinations)
        
        if is_chiplet:
            # carbon = np.zeros((len(combinations), len(area_list)))
            carbon = np.zeros((len(combinations),16))
            for n, comb in enumerate(combinations): 
                cpa = self.get_carbon_per_area(comb)
                defect_density = self.get_defect_rate(comb) 
                yields = []               
                wastage_extra_cfp = [] 
                for i, c in enumerate(comb): 
                    yields.append(self.yield_calc(area_list[i],defect_density[i]))
                    wastage_extra_cfp.append(self.waste_carbon_per_die(diameter=450, chip_area=area_list[i], cpa_factors=cpa[i]))
                print("yield :", yields)
                carbon = cpa*np.array(area_list) / yields + wastage_extra_cfp   # in g
                carbon = ((1-self.rcy_mat_frac)*carbon) + (self.rcy_mat_frac*carbon*self.rcy_cpa_frac)
                # breakpoint()
                design_carbon_per_chiplet, design_carbon = self.design_costs(core_area_list, 8,10,700,comb, is_chiplet)
                # breakpoint()
                package_c, router_c, design_carbon_package, router_a = self.package_costs(area_list, comb, area_list, True, is_chiplet, 700, self.interposer_area, noc_area_list)
                print(package_c, router_c, design_carbon_package, router_a)
                package_carbon = package_c * 1 # + router_c   # in g
                # breakpoint()
            print("design_carbon: ", design_carbon)
            #total_carbon = carbon.sum(axis=1)
            carbon = carbon + package_carbon/len(carbon)
            # carbon = ((1-self.rcy_mat_frac)*carbon) + (self.rcy_mat_frac*carbon*self.rcy_cpa_frac)
            print("carbon afteer:", carbon)
            total_carbon = carbon.sum()
            cdes = design_carbon_per_chiplet.sum() + design_carbon_package.sum()
            # breakpoint()
            print("carbon: ", carbon)
        else: 
            cpa = self.get_carbon_per_area([self.carbonparam.technology_node])
            print("cpa: ", cpa)
            defect_density = self.get_defect_rate([self.carbonparam.technology_node])
            print("defect_desnity: ", defect_density)
            total_area = sum(area_list)
            yields = self.yield_calc(total_area, defect_density[0])
            print("yields",yields)
            wastage_extra_cfp = self.waste_carbon_per_die(diameter=450, chip_area=total_area, cpa_factors=cpa[0])
            print("wastage_extra_cfp:", wastage_extra_cfp)
            area = np.array(total_area)
            carbon = cpa*area/yields + wastage_extra_cfp
            print(carbon)
            design_carbon_per_chiplet, design_carbon = self.design_costs(area_list, 8,10,700,combinations[0], is_chiplet)
            print("---------design calculate over-----------")
            print("design_carbon_per_chiplet:", design_carbon_per_chiplet)
            package_c, router_c, design_carbon_package, router_a = self.package_costs(area_list, combinations[0], area_list, True, is_chiplet, 700, self.interposer_area)
            print(package_c, router_c, design_carbon_package, router_a)
            total_carbon = carbon +  package_c * 1 + router_c
            # breakpoint()
            cdes = design_carbon_per_chiplet.sum()
        app_dev_c = self.app_cfp(power_per_core=10, num_core=8, Carbon_per_kWh=700,
                            Na=0,Ns=0,fe_time=0.2,be_time=0.05,config_time=0)
    
        #Recycle CFP
        eol_c = self.end_cfp(cpa_dis_p_Ton=10, cpa_rcy_p_Ton=2, dis_frac=1, weight_p_die=2)
        print("design_carbon_per_chiplet:", design_carbon_per_chiplet)
        # cdes = design_carbon_per_chiplet.sum() + design_carbon_package.sum()
        #cmfg = total_carbon

        if is_chiplet: 
            cmfg = carbon
            print("cmfg:", cmfg)
        else: 
            cmfg = np.array(total_carbon)
            print("cmfg:", cmfg)
        ceol = eol_c
        capp = app_dev_c

        cdes = cdes/1000
        cmfg = cmfg/1000
        ceol = ceol/1000
        capp = capp/1000

        des_c, mfg_c, eol_c, app_c = self.total_cfp_gen(1,cdes,cmfg,1,1, ceol, capp,1)
        # print("des_c, mfg_c, eol_c, app_c:", des_c, mfg_c, eol_c, app_c)
        # emb_c = des_c + mfg_c + eol_c + app_c     
        if is_chiplet:
            # emb_c = dict(zip(core_id_list,[mfg_c+(des_c+eol_c+app_c)/len(core_id_list)]))
            # emb_c = dict(zip(core_id_list, mfg_c+(des_c+eol_c+app_c)/len(core_id_list)))
            mfg_c = (sum(mfg_c)+(des_c+eol_c+app_c))/len(core_id_list)
            emb_c = dict.fromkeys(core_id_list, mfg_c)
            print("core id list: ", core_id_list)
        else:
            value = float((des_c + mfg_c + eol_c + app_c)/len(core_id_list))
            emb_c = dict.fromkeys(core_id_list, value)
            # emb_c = dict(zip(core_id_list,value_list))
        print("emb_c: ", emb_c)
        print("embc_core: ", emb_c[0])
        # breakpoint()
        return emb_c

    def design_costs(self, areas, Transistors_per_gate,Power_per_core,Carbon_per_kWh, technology_node, is_chiplet):
        """
        return result is in kg
        """
        print("--------------inside design costs:-------------------")
        transistor_density = self.data["transistor_scaling"]["Transistors_per_mm2"]
        print("transistor_density: ", transistor_density)
        gates_design_time = self.data["gates_per_hr_per_core"]["Gates_per_hr_per_core"]
        nodes = self.data["technology_node"]["technology_node"][:-1]
        print("nodes: ", nodes)
        if is_chiplet:
            if technology_node[0] in nodes: 
                transistors = areas[0]*transistor_density[nodes.index(technology_node[0])]
            else: 
                transistors = areas[0]*np.interp(technology_node[0], nodes, 
                                    transistor_density)
        else:
            if technology_node[0] in nodes: 
                print(transistor_density[nodes.index(technology_node[0])])
                transistors = sum(areas)*transistor_density[nodes.index(technology_node[0])]
                print("transistor number:", transistors)
            else:
                print(np.interp(technology_node[0], nodes, transistor_density))
                print(sum(areas))
                transistors = sum(areas)*np.interp(technology_node[0], nodes, 
                                    transistor_density)
                print("transistor number:", transistors)
        gates = transistors/Transistors_per_gate

        if technology_node[0] in nodes: 
            CPU_core_hours = gates/gates_design_time[nodes.index(technology_node[0])]
        else: 
            CPU_core_hours = gates/np.interp(technology_node[0], nodes, 
                                    gates_design_time)
            print("CPU_core_hours:", CPU_core_hours)
        total_energy = Power_per_core*CPU_core_hours/1000  #in kWh
        design_carbon = Carbon_per_kWh * total_energy
        design_carbon_per_chiplet = design_carbon*100/1e5
        return design_carbon_per_chiplet, design_carbon
    
    def package_costs(self, area_list, technology_node, new_area, return_router_area, is_chiplet, carbon_per_kwh, interposer_area_fix, router_area_fix = 0):
        package_carbon = 0 
        router_carbon = 0
        router_design = 0
        router_area = 0
        package_param = self.data["package_param"]
        bonding_yield = package_param["bonding_yield"]
        new_area = [float(x) for x in new_area]
        if is_chiplet:
            num_chiplets = len(new_area)
            # interposer_area = sum(area_list)
            # interposer_area = len(new_area) * area_list[0]/6 * 62.67361111111111
            interposer_area = interposer_area_fix
            num_if = len(self.accelerator.communication_manager.pair_links)
            interposer_carbon = self.package_mfg_carbon(package_param["interposer_node"], [interposer_area])
            logic_scaling = self.data["logic_scaling"]
            sram_scaling = self.data["sram_scaling"]
            analog_scaling = self.data["analog_scaling"]
            scalings = list(zip(logic_scaling["area"], analog_scaling["area"], sram_scaling["area"]))
            beolVfeol = self.data["beolVfeol_scaling"]["beolVfeol"]
            # breakpoint()
            nodes = self.data["technology_node"]["technology_node"]
            package = self.carbonparam.package_type
            if package == "active": 
                if router_area_fix != 0:
                    router_area = router_area_fix
                else:
                    router_area = 4.47 * num_chiplets
                    router_area = interposer_carbon * router_area/interposer_area
                router_design = 0 
                package_carbon = interposer_carbon * beolVfeol[-1]
            elif package == "3D":
                dims = np.sqrt(np.array(area_list, dtype=np.float64))
                num_tsv_1d = np.floor(dims/package_param["tsv_pitch"])
                overhead_3d = (num_tsv_1d**2) * (package_param["tsv_size"]**2)
                area_3d = area_list + overhead_3d
                carbon3d= self.package_mfg_carbon(technology_node,area_3d)
                carbon2d= self.package_mfg_carbon(technology_node,area_list)
                package_carbon = np.sum(carbon3d-carbon2d)
                router_area = []
                if router_area_fix != 0: 
                    router_area = router_area_fix
                else:
                    for index in range(len(area_list)):
                        router_area.append(0.33/scalings[2][index])
                router_carbon= self.package_mfg_carbon(technology_node, router_area)
                router_carbon= np.sum(router_carbon)
                router_design = 0
                bonding_yield = bonding_yield**num_chiplets
            elif package in ['passive', 'RDL', 'EMIB']: 
                router_area = []
                if router_area_fix != 0: 
                    router_area = router_area_fix
                else:
                    for index in range(len(area_list)):
                        router_area.append(0.33/scalings[2][0])
                # router_carbon= self.package_mfg_carbon(technology_node, router_area)
                cpa = self.get_carbon_per_area(technology_node) 
                defect_density = self.get_defect_rate(technology_node)
                # new_area = area_scaling(router_area,technology_node)
                new_area = router_area
                yields = []
                wastage_extra_cfp = []
                # if ~(np.all(np.array(technology_node) == technology_node[0])): 
                #     for i, c in enumerate(technology_node):   
                #         yields.append(self.yield_calc(new_area[i], defect_density[i]))
                #         wastage_extra_cfp.append(self.waste_carbon_per_die(diameter=450,chip_area=new_area[i],cpa_factors=cpa[i]))
                # else: 
                #     yields = self.yield_calc(sum(new_area), defect_density[0])
                #     wastage_extra_cfp = self.waste_carbon_per_die(diameter=450,chip_area=sum(new_area),cpa_factors=cpa[0])
                #     wastage_extra_cfp = np.array(wastage_extra_cfp)
                #     wastage_extra_cfp = (wastage_extra_cfp * router_area) / sum(router_area)
                for i, c in enumerate(technology_node):   
                    yields.append(self.yield_calc(new_area[i], defect_density[i]))
                    wastage_extra_cfp.append(self.waste_carbon_per_die(diameter=450,chip_area=new_area[i],cpa_factors=cpa[i]))
                router_carbon = cpa*np.array(new_area) / yields# + wastage_extra_cfp
                # breakpoint()
                design_carbon_per_chiplet, router_design = self.design_costs(new_area, 8, 10, 700, technology_node, True)
                router_carbon, router_design = np.sum(router_carbon), np.sum(design_carbon_per_chiplet)
                if package == 'passive':
                    package_carbon = interposer_carbon* beolVfeol[nodes.index(package_param["interposer_node"])]
                elif package == 'RDL':
                    package_carbon = interposer_carbon* beolVfeol[nodes.index(package_param["interposer_node"])]
                    package_carbon *= package_param["RDLLayers"]/package_param["numBEOL"] 
                    # breakpoint()  
                elif package == 'EMIB':
                    emib_area =  [5*5]*num_if
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
        
    def package_mfg_carbon(self, technology_node, interposer_area): 
        cpa = self.get_carbon_per_area([technology_node]) 
        defect_density = self.get_defect_rate([technology_node])
        new_area = interposer_area
        # breakpoint()
        total_area = sum(new_area)
        yields = []
        wastage_extra_cfp = []
        yields = self.yield_calc(sum(new_area), defect_density[0]/4)
        """
        wastage_extra_cfp = self.waste_carbon_per_die(diameter=450,chip_area=sum(new_area),cpa_factors=cpa[0])
        wastage_extra_cfp = (wastage_extra_cfp * new_area[0]) / sum(new_area)
        """
        
        interposer_carbon = cpa*np.array(new_area) / yields #  + wastage_extra_cfp
        interposer_carbon = ((1-self.rcy_mat_frac)*interposer_carbon) + (self.rcy_mat_frac*interposer_carbon*self.rcy_cpa_frac)
        return interposer_carbon
    
    ###############################################
    #Programming CFP 

    def app_cfp(self, power_per_core,num_core,Carbon_per_kWh,Na,Ns,fe_time,be_time,config_time):
        prog_time = ((Na*(fe_time+be_time)) + (Ns*config_time)) * 24*30 #Converting to hrs from months 
        program_energy = power_per_core*num_core*prog_time/1000 #in kWh
        prog_cfp = program_energy*Carbon_per_kWh
        return prog_cfp
    
    ###############################################

    def end_cfp(self, cpa_dis_p_Ton, cpa_rcy_p_Ton, dis_frac, weight_p_die):
        cpa_dis_p_gm = cpa_dis_p_Ton/1000 
        cpa_rcy_p_gm = cpa_rcy_p_Ton/1000
        dis_cfp = cpa_dis_p_gm*weight_p_die
        rcy_cfp = cpa_rcy_p_gm*weight_p_die
        eol_cfp = (dis_frac*dis_cfp)-((1-dis_frac)*rcy_cfp)
        return eol_cfp
    
    ###################

    def total_cfp_gen(self,num_des,des_c_pu,mfg_c_pu,n_fpga,vol,eol_c_pu,app_c_tot,dc):
        design_cfp_total = num_des*des_c_pu
        mfg_cfp_total = n_fpga*(mfg_c_pu*num_des*vol)
        eol_cfp_total = n_fpga*(eol_c_pu*num_des*vol)
        app_cfp_total = app_c_tot
        return design_cfp_total,mfg_cfp_total,eol_cfp_total,app_cfp_total


    # def is_leaf(self) -> bool:
    #     return True