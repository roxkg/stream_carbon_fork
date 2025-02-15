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
    ):
        self.CI_op = CI_op
        self.CI_em = CI_em
        self.lifetime = lifetime

    
        