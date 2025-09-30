import torch.nn as nn 
from .utils_function import conv3x3


class GateNetwork(nn.Module):
    def __init__(self,in_dim, mid_dim ,num_adapter,considered_channels = -1, depth = 1):
        super().__init__()
        self.num_adapter = num_adapter
        self.considered_channels = considered_channels
        self.in_dim = in_dim 

        self.depth = depth 


        self.gate_net = nn.ModuleList([]) 
        in_dim = self.in_dim 
        self.middle_dim = mid_dim
        for i in range(depth):



            """
            params =  OrderedDict([
                ( "Gate_conv" + str(i),conv3x3(in_dim,middle_dim , kernel_size = 3,stride  = 1)),
                ('Gate_gelu' + str(i),nn.GELU()),
                ("Gate_MaxPool2d" + str(i),nn.MaxPool2d(2, 2))])
                    
                    
            single_layer =  nn.Sequential(params)
            """
            single_layer =  nn.ModuleList([
                conv3x3(in_dim,self.middle_dim , kernel_size = 3,stride  = 1),
                nn.GELU(),
                nn.MaxPool2d(2, 2)])
                    
                    
            

            self.gate_net.append(single_layer)


            in_dim = self.middle_dim 
            
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.Linear = nn.Linear(self.middle_dim, self.num_adapter)


    
    def forward(self,x):
        for conv, gelu, pool in self.gate_net:
            x = conv(x)
            x = gelu(x)
            x = pool(x) 
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.Linear(x)
        return logits
    

    def print_information_gate(self):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(" the number of paramaters of the gate is")
        print(sum(p.numel() for p in self.gate_net.parameters())," ",sum(p.numel() for p in self.Linear.parameters()))
        print(sum(p.numel() for p in self.gate_net.parameters()) + sum(p.numel() for p in self.Linear.parameters()))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


