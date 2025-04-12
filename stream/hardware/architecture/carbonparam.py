class CarbonParam: 
    """
    The CarbonParam class holds carbon intensity, lifetime, technology node and other parameteres
    related with accelerator itself. This is used in carbon efficiency optimization direction. 
    """

    def __init__(
        self,
        CI_op: int, 
        CI_em: int, 
        lifetime: int, 
        frequency: float,
        technology_node: int,
        package_type: str
    ):
        self.CI_op = CI_op
        self.CI_em = CI_em
        self.lifetime = lifetime
        self.frequency = frequency
        self.technology_node = technology_node
        self.package_type = package_type
        self.mfg_carbon = 0
        self.design_carbon = 0
        self.package_carbon = 0
        self.cemb = None
        
    def set_cemb(self, cemb):
        self.cemb = cemb

    
        