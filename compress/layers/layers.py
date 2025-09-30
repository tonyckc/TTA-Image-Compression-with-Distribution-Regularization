# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch import Tensor
import torch
import torch.nn as nn
from .win_attention import WinBaseAttentionAdapter
from compress.adaptation.adapter import Adapter
#from .wacn_adapter import define_adapter, init_adapter_layer

from collections import OrderedDict
from .utils_function import  conv3x3, subpel_conv3x3


from .base import Win_noShift_Attention, AttentionBlock, ResidualBlock, ResidualBlockUpsample
from .gdn import GDN 




class ZeroDeconvLayer(nn.Module):
    def __init__(self,  out_channels, kernel_size, stride ):
        super().__init__()
        self.out_channels = out_channels 
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding =  self.kernel_size // 2
        self.output_padding = self.stride -1

    
    def forward(self,x):
        bs,_,w,h = x.shape



        w_out = (w - 1)*self.stride - 2*self.padding + self.kernel_size + self.output_padding
        h_out = (h - 1)*self.stride - 2*self.padding + self.kernel_size + self.output_padding

        y = torch.zeros((bs,self.out_channels,w_out, h_out )).to("cuda")
        return y


class ResidualMultipleAdaptersDeconv(nn.Module):
    def __init__(self, in_channels, 
                    out_channels, 
                    kernel_size=5, 
                    stride=2, 
                    mean = 0, 
                    standard_deviation = 0.00,
                    initialize = "gaussian", 
                    num_adapter = 3, 
                    aggregation = "weighted", 
                    threshold = 0,
                    skipped = False):
        super().__init__()

        self.original_model_weights = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1,
            padding=kernel_size // 2,
        )


        self.mean = mean 
        self.standard_deviation = standard_deviation
        self.aggregation = aggregation
        self.threshold = threshold
        self.num_adapter = num_adapter


        self.adapters = nn.ModuleList([]) 
        for i in range(num_adapter):
            if i == 0 and skipped:
                name_ad = "adapter_skipped" + str(i)
            else:
                name_ad = "adapter_" + str(i)



            if skipped and i == 0:   
                params = OrderedDict([(name_ad,ZeroDeconvLayer(out_channels,kernel_size,stride))])  
            else:
                params = OrderedDict([(name_ad,nn.ConvTranspose2d(in_channels, out_channels,  kernel_size = kernel_size, stride = stride, output_padding = stride -1, padding = kernel_size // 2))]) 
            adapter =  nn.Sequential(params)
            if initialize == "gaussian":
                adapter.apply(self.initialization)
            self.adapters.append(adapter)


    def initialization(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear) or isinstance(m, nn.ConvTranspose2d) :
            torch.nn.init.zeros_(m.weight) #nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias) 


    def extract_prob_distribution(self,gate_prob, oracle):
        if oracle is None:
            argmax = torch.argmax(gate_prob, dim = 1)
        else:
            #print("the oracle is --> ",oracle)
            argmax = oracle # prendo le classi dell'oracolo

    
        if self.aggregation == "top1" or oracle is not None:
            #one ho encoding 
            one_hot_matrix = torch.eye(self.num_adapter)
            one_hot_encoded = one_hot_matrix[argmax]
            gate_prob = one_hot_encoded.to("cuda")
        elif self.aggregation == "weighted" :

            gate_prob = torch.where(gate_prob < self.threshold, torch.zeros_like(gate_prob), gate_prob).to("cuda")


        else: 
            raise ValueError("Per ora non ho implementato altro!!!!!!!")

        return gate_prob


    def forward(self,x, gate_prob, oracle):

        x_conv = self.original_model_weights(x)

        gate_prob = self.extract_prob_distribution(gate_prob, oracle)


            
            
        summed_out = x.unsqueeze(1).repeat(1,self.num_adapter,1,1,1) # [16,3,192,18,24] #torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) 
        ad_summed_out = torch.stack([self.adapters[i](summed_out[:,i,:,:,:]) for i in range(self.num_adapter)], dim = 1)
        ad_summed_out = ad_summed_out*gate_prob[:,:,None,None,None]
     
        x_adapt = torch.sum(ad_summed_out, dim =1) #[16,192,18,24]
        return x_conv + x_adapt 


    











class Win_noShift_Attention_Multiple_Adapter(Win_noShift_Attention):

    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=8,
        shift_size=0,
        dim_adapter: int = 1,
        groups: int = 1,
        aggregation: str = "top1",
        stride: int = 1,
        num_adapter: int = 3,
        kernel_size: int = 1,
        std: float = 0.01,
        mean: float = 0.00,
        bias: bool = False, 
        padding: int = 0,
        name: list =[],
        type_adapter: str = "singular",
        position: str = "res",
        threshold: int = 0
    ):
        """Win_noShift_Attention with multiple adapters.
        """
        super().__init__(dim, num_heads, window_size, shift_size)
        self.aggregation = aggregation
        self.num_adapter = num_adapter
        self.threshold = threshold
        self.position = position
        self.adapters = nn.ModuleList([]) 
        for i in range(num_adapter):

            if len(name) == 0:
                name_ad = "AttentionAdapter_" + str(i)
            else: 
                name_ad = "AttentionAdapter_" + name[i]


            params = OrderedDict([
                    (name_ad,Adapter(dim,
                                    dim, 
                                    dim_adapter=dim_adapter[i], 
                                    groups=groups, 
                                    stride = stride[i], 
                                    padding = padding[i], 
                                    standard_deviation= std,
                                    mean=mean,
                                    kernel_size= kernel_size[i], 
                                    bias = bias,
                                    name = name_ad, 
                                    type_adapter= type_adapter[i]))])
                                    
            adapter =  nn.Sequential(params)
            self.adapters.append(adapter)



    def aggregate_adapters(self, out, gate_prob, oracle):
        #gate_prob = nn.Softmax(gate_output) # estraggo le probabilità delle singole classi

        if oracle is None:
            argmax = torch.argmax(gate_prob, dim = 1)
        else:
            #print("the oracle is --> ",oracle)
            argmax = oracle # prendo le classi dell'oracolo

    
        if self.aggregation == "top1" or oracle is not None:
            #one ho encoding 
            one_hot_matrix = torch.eye(self.num_adapter)
            one_hot_encoded = one_hot_matrix[argmax]
            gate_prob = one_hot_encoded.to("cuda")
            #if oracle is not None:
                #print("the shape of gate_prob is: ",gate_prob.shape)
        elif self.aggregation == "weighted":
            gate_prob = torch.where(gate_prob < self.threshold, torch.zeros_like(gate_prob), gate_prob).to("cuda")
        else: 
            raise ValueError("Per ora non ho implementato altro!!!!!!!")
        


        summed_out = out.unsqueeze(1).repeat(1,self.num_adapter,1,1,1) # [16,3,192,18,24] #torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) 
        ad_summed_out = torch.stack([self.adapters[i](summed_out[:,i,:,:,:]) for i in range(self.num_adapter)], dim = 1)
        ad_summed_out = ad_summed_out*gate_prob[:,:,None,None,None]
        ad_summed_out = torch.sum(ad_summed_out, dim =1) #[16,192,18,24]
        if self.position == "res":
            return  out  + ad_summed_out # torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) # [BS, num_adapter, h, w]
        else: 
            return ad_summed_out #torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1)




        




    def forward(self, x, gate_prob, oracle = None):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        return self.aggregate_adapters(out, gate_prob, oracle) + identity







class AttentionBlockWithMultipleAdapters(AttentionBlock):
    def __init__(self, N: int,
                        dim_adapter: int = 1,
                        groups: int = 1,
                        aggregation: str = "top1",
                        stride: int = 1,
                        num_adapter: int = 3,
                        kernel_size: int = 1,
                        std: float = 0.01,
                        mean: float = 0.00,
                        bias: bool = False, 
                        padding: int = 0,
                        name: list =[],
                        type_adapter: str = "singular",
                        position: str = "res",
                        threshold: int = 0):
        super().__init__(N)

        self.aggregation = aggregation
        self.num_adapter = num_adapter
        self.threshold = threshold
        self.position = position
        self.adapters = nn.ModuleList([]) 
        for i in range(num_adapter):

            if len(name) == 0:
                name_ad = "AttentionAdapter_" + str(i)
            else: 
                name_ad = "AttentionAdapter_" + name[i]


            params = OrderedDict([
                    (name_ad,Adapter(N,
                                    N, 
                                    dim_adapter=dim_adapter[i], 
                                    groups=groups, 
                                    stride = stride[i], 
                                    padding = padding[i], 
                                    standard_deviation= std,
                                    mean=mean,
                                    kernel_size= kernel_size[i], 
                                    bias = bias,
                                    name = name_ad, 
                                    type_adapter= type_adapter[i]))])
                                    
            adapter =  nn.Sequential(params)
            self.adapters.append(adapter)





    def aggregate_adapters(self, out, gate_prob, oracle):
        #gate_prob = nn.Softmax(gate_output) # estraggo le probabilità delle singole classi

        if oracle is None:
            argmax = torch.argmax(gate_prob, dim = 1)
        else:
            #print("the oracle is --> ",oracle)
            argmax = oracle # prendo le classi dell'oracolo

    
        if self.aggregation == "top1" or oracle is not None:
            #one ho encoding 
            one_hot_matrix = torch.eye(self.num_adapter)
            one_hot_encoded = one_hot_matrix[argmax]
            gate_prob = one_hot_encoded.to("cuda")
        elif self.aggregation == "weighted":
            gate_prob = torch.where(gate_prob < self.threshold, torch.zeros_like(gate_prob), gate_prob).to("cuda")
        else: 
            raise ValueError("Per ora non ho implementato altro!!!!!!!")
        


        summed_out = out.unsqueeze(1).repeat(1,self.num_adapter,1,1,1).to("cuda") # [16,3,192,18,24] #torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) 
        ad_summed_out = torch.stack([self.adapters[i](summed_out[:,i,:,:,:]) for i in range(self.num_adapter)], dim = 1)
        ad_summed_out = ad_summed_out*gate_prob[:,:,None,None,None].to("cuda")
        ad_summed_out = torch.sum(ad_summed_out, dim =1) #[16,192,18,24]
        if self.position == "res":
            return  out  + ad_summed_out # torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) # [BS, num_adapter, h, w]
        else: 
            return ad_summed_out #torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1)




    def forward(self, x, gate_prob, oracle) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity # introdurre qua gli adapters
        return self.aggregate_adapters(out, gate_prob, oracle) + identity
    





class ResidualBlockMultipleAdapters(ResidualBlock):

    def __init__(self, 
                 in_ch,
                out_ch,
                mean = 0, 
                standard_deviation = 0.00,
                initialize = "gaussian", 
                num_adapter = 3, 
                name = [], 
                aggregation = "weighted", 
                threshold = 0):
        super().__init__(in_ch,out_ch) #original_model
        



        self.mean = mean 
        self.standard_deviation = standard_deviation
        self.aggregation = aggregation
        self.threshold = threshold
        self.num_adapter = num_adapter


        self.adapters = nn.ModuleList([]) 
        for i in range(num_adapter):
            if len(name) == 0:
                name_ad = "residualadapter_" + str(i)
            else:
                name_ad = "residualadapter_" + name[i]
            params = OrderedDict([(name_ad, conv3x3(out_ch, out_ch))]) 
            adapter =  nn.Sequential(params)
            if initialize == "gaussian":
                adapter.apply(self.initialization)
            self.adapters.append(adapter)


    def initialization(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear) or isinstance(m, nn.ConvTranspose2d) :
            torch.nn.init.zeros_(m.weight) #nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias) 



    def extract_prob_distr(self, gate_prob, oracle):
        if oracle is None:
            argmax = torch.argmax(gate_prob, dim = 1)
        else:
            #print("the oracle is --> ",oracle)
            argmax = oracle # prendo le classi dell'oracolo

    
        if self.aggregation == "top1" or oracle is not None:
            #one ho encoding 
            one_hot_matrix = torch.eye(self.num_adapter)
            one_hot_encoded = one_hot_matrix[argmax]
            gate_prob = one_hot_encoded.to("cuda")
        elif self.aggregation == "weighted" :
            gate_prob = torch.where(gate_prob < self.threshold, torch.zeros_like(gate_prob), gate_prob).to("cuda")

        else: 
            raise ValueError("Per ora non ho implementato altro!!!!!!!")  

        return gate_prob     

    def forward(self, x, gate_prob, oracle):

        identity = x

        out_1 = self.conv1(x)
        out_1 = self.leaky_relu(out_1)
        out_2 = self.conv2(out_1)
        out_2= self.leaky_relu(out_2)

 
        gate_prob = self.extract_prob_distr(gate_prob, oracle)


        summed_out = out_1.unsqueeze(1).repeat(1,self.num_adapter,1,1,1) # [16,3,192,18,24] #torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) 
        ad_summed_out = torch.stack([self.adapters[i](summed_out[:,i,:,:,:]) for i in range(self.num_adapter)], dim = 1)
        ad_summed_out = ad_summed_out*gate_prob[:,:,None,None,None]
         
        out_adapt = torch.sum(ad_summed_out, dim =1) #[16,192,18,24]
        out_adapt = self.leaky_relu(out_adapt)
        out = out_2 + out_adapt + identity
        return out
    


class ResidualBlockUpsampleMultipleAdapters(ResidualBlockUpsample):
    def __init__(self, in_ch: int,
                out_ch: int,
                upsample: int = 2,
                mean = 0, 
                standard_deviation = 0.00,
                initialize = "gaussian", 
                num_adapter = 3, 
                name = [], 
                aggregation = "weighted", 
                threshold = -1,
                ):

        super().__init__( in_ch = in_ch, out_ch = out_ch, upsample = upsample)

        self.threshold = threshold
        self.mean = mean 
        self.standard_deviation = standard_deviation
        self.aggregation = aggregation
        self.num_adapter = num_adapter


        self.adapters = nn.ModuleList([]) 
        for i in range(num_adapter):
            if len(name) == 0:
                name_ad = "residualadapter_" + str(i)
            else:
                name_ad = "residualadapter_" + name[i]
            params = OrderedDict([(name_ad, conv3x3(out_ch, out_ch))]) 
            adapter =  nn.Sequential(params)
            if initialize == "gaussian":
                adapter.apply(self.initialization)
            self.adapters.append(adapter)
        
    def initialization(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear) or isinstance(m, nn.ConvTranspose2d) :
            torch.nn.init.zeros_(m.weight) #nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)    


    def extract_prob_distr(self, gate_prob, oracle):
        if oracle is None:
            argmax = torch.argmax(gate_prob, dim = 1)
        else:
            #print("the oracle is --> ",oracle)
            argmax = oracle # prendo le classi dell'oracolo

    
        if self.aggregation == "top1" or oracle is not None:
            #one ho encoding 
            one_hot_matrix = torch.eye(self.num_adapter)
            one_hot_encoded = one_hot_matrix[argmax]
            gate_prob = one_hot_encoded.to("cuda")
        elif self.aggregation == "weighted" :
            gate_prob = torch.where(gate_prob < self.threshold, torch.zeros_like(gate_prob), gate_prob).to("cuda")

        else: 
            raise ValueError("Per ora non ho implementato altro!!!!!!!")  

        return gate_prob   


    def forward(self,x, gate_prob, oracle):
        identity = x


        

        out_1 = self.subpel_conv(x)
        out_1 = self.leaky_relu(out_1)
        out_2 = self.conv(out_1)
        out_2 = self.igdn(out_2)
        identity = self.upsample(x)


        gate_prob = self.extract_prob_distr(gate_prob, oracle)



        summed_out = out_1.unsqueeze(1).repeat(1,self.num_adapter,1,1,1) # [16,3,192,18,24] #torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) 
        ad_summed_out = torch.stack([self.adapters[i](summed_out[:,i,:,:,:]) for i in range(self.num_adapter)], dim = 1)
        ad_summed_out = ad_summed_out*gate_prob[:,:,None,None,None]
         
        out_adapt = torch.sum(ad_summed_out, dim =1) #[16,192,18,24]

        out_adapt = self.leaky_relu(out_adapt)


        out = out_2 + out_adapt +  identity
        return out




class subpel_conv3x3MultipleAdapters(nn.Module):

    def __init__(self, in_ch: int,
                out_ch: int,
                r: int = 2,
                mean = 0, 
                standard_deviation = 0.00,
                initialize = "gaussian", 
                num_adapter = 3, 
                name = [], 
                aggregation = "weighted", 
                threshold = -1
                ):
        
        super().__init__()
        self.original_model_weights = nn.Sequential(nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r))

        self.threshold = threshold
        self.mean = mean 
        self.standard_deviation = standard_deviation
        self.aggregation = aggregation
        self.num_adapter = num_adapter


        self.adapters = nn.ModuleList([]) 
        for i in range(num_adapter):
            if len(name) == 0:
                name_ad = "subpeladapter_" + str(i)
            else:
                name_ad = "subpeladapter_" + name[i]
            params = OrderedDict([
                                (name_ad + "_conv2d", nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1)),
                                (name_ad + "_pixelshuffle",nn.PixelShuffle(r))
                                ]) 
            adapter =  nn.Sequential(params)
            if initialize == "gaussian":
                adapter.apply(self.initialization)
            self.adapters.append(adapter)
        
    def initialization(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear) or isinstance(m, nn.ConvTranspose2d) :
            torch.nn.init.zeros_(m.weight) #nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)    



    def extract_prob_distr(self, gate_prob, oracle):

        if oracle is None:
            argmax = torch.argmax(gate_prob, dim = 1)
        else:
            #print("the oracle is --> ",oracle)
            argmax = oracle # prendo le classi dell'oracolo

    
        if self.aggregation == "top1" or oracle is not None:
            #one ho encoding 
            one_hot_matrix = torch.eye(self.num_adapter)
            one_hot_encoded = one_hot_matrix[argmax]
            gate_prob = one_hot_encoded.to("cuda")
        elif self.aggregation == "weighted" :

            gate_prob = torch.where(gate_prob < self.threshold, torch.zeros_like(gate_prob), gate_prob).to("cuda")


        else: 
            raise ValueError("Per ora non ho implementato altro!!!!!!!")
        
        return gate_prob



    def forward(self,x, gate_prob, oracle):

        x_conv = self.original_model_weights(x)

        gate_prob = self.extract_prob_distr(gate_prob, oracle)



            
            
        summed_out = x.unsqueeze(1).repeat(1,self.num_adapter,1,1,1) # [16,3,192,18,24] #torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) 
        ad_summed_out = torch.stack([self.adapters[i](summed_out[:,i,:,:,:]) for i in range(self.num_adapter)], dim = 1)
        ad_summed_out = ad_summed_out*gate_prob[:,:,None,None,None]
         
        x_adapt = torch.sum(ad_summed_out, dim =1) #[16,192,18,24]
            
        #x_adapt = torch.sum(torch.stack([self.adapters[i](x)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1)
        return x_conv + x_adapt 





##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################


class Win_noShift_Attention_Adapter(Win_noShift_Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=8,
        shift_size=0,
        dim_adapter: int = 1,
        groups: int = 1,
        position: str = "res_last",
        stride: int = 1,

        kernel_size: int = 1,
        std: float = 0.01,
        mean: float = 0.00,
        bias: bool = False, 
        padding: int = 0,
        name: str = "",
        type_adapter: str = "singular"
    ):
        """Win_noShift_Attention with adapters.

        Args:
            dim_adapter (int): dimension for the intermediate feature of the adapter.
            groups (int): number of groups for the adapter. Default: 1. if groups=dim, the adapter become channel-wise multiplication.
        """
        super().__init__(dim, num_heads, window_size, shift_size)
        self.position = position
        if self.position in ("res_last","last"):
            self.adapter = Adapter(dim, 
                                   dim, 
                                   dim_adapter=dim_adapter, 
                                   groups=groups, 
                                   stride = stride, 
                                   padding = padding, 
                                   standard_deviation= std,
                                   mean=mean,
                                   kernel_size= kernel_size, 
                                   bias = bias,
                                   name = name, 
                                   type_adapter= type_adapter

                                   )
            #self.adapter.apply(init_adapter_layer)
        elif self.position in {"attn", "attnattn"}:
            self.conv_b[0] = WinBaseAttentionAdapter(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                dim_adapter=dim_adapter,
                groups=groups,
                position=self.position,
            )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)

        if self.position == "res_last" :
            # modify output by adapters
            out = out + self.adapter(out)
        elif self.position == "last":
            out = self.adapter(out)


        out += identity
        return out


class ResidualAdapterDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, mean = 0, standard_deviation = 0.00, initialize = "gaussian"):
        super().__init__()
        self.original_model_weights = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1,
            padding=kernel_size // 2,
        )
        

        params = OrderedDict([("adapter_transpose_conv1",nn.ConvTranspose2d(in_channels, out_channels,  kernel_size = kernel_size, stride = stride, output_padding = stride -1, padding = kernel_size // 2))])
                    
                    
        self.Adapterconv =  nn.Sequential(params)
        self.mean = mean 
        self.standard_deviation = standard_deviation
        
        if initialize == "gaussian": #init
            self.Adapterconv.apply(self.initialization)
    

    def initialization(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear) or isinstance(m, nn.ConvTranspose2d) :

            torch.nn.init.zeros_(m.weight) #nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)       
    
    def forward(self, x):
        x_adapt = self.Adapterconv(x) 
        x_conv = self.original_model_weights(x)
        return x_conv + x_adapt
