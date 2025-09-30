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


from compress.models import  WACNN , WACNNGateAdaptive,  Cheng2020AttnAdapter 
from compressai.models import Cheng2020Attention
import torch
from compressai.zoo import *
from compressai.zoo import    image_models
from .pretrained import load_pretrained as load_state_dict

models = {
    "base": WACNN,
    "gate": WACNNGateAdaptive,
    "cheng":Cheng2020AttnAdapter,
    "base_cheng": Cheng2020Attention
}


def from_state_dict(cls, state_dict):
    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net



def rename_key_for_adapter(key, stringa, nuova_stringa):
    if key.startswith(stringa):
        key = nuova_stringa # nuova_stringa  + key[6:]
    return key

def get_gate_model(args,num_adapter, device):
    pret_checkpoint = args.pret_checkpoint + "/" + args.quality + "/model.pth"

    if args.name_model == "cheng":
        
        
        qual = int(args.quality[1])
        modello_base =  image_models["cheng2020-attn"](quality=qual, metric="mse", pretrained=True, progress=False)
        modello_base.update()
        modello_base.to(device)   

        state_dict = modello_base.state_dict() 


        state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.9.original_model_weights.0.weight", stringa = "g_s.9.0.weight" ): v for k, v in state_dict.items()}
        state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.9.original_model_weights.0.bias", stringa = "g_s.9.0.bias" ): v for k, v in state_dict.items()}

        net = models["cheng"](N=args.N,
                                dim_adapter_attn = args.dim_adapter_attn,
                                stride_attn = args.stride_attn,
                                kernel_size_attn = args.kernel_size_attn,
                                padding_attn = args.padding_attn,
                                type_adapter_attn = args.type_adapter_attn,
                                position_attn = args.position_attn,
                                num_adapter = num_adapter,
                                aggregation = args.aggregation,
                                std = args.std,
                                mean = args.mean,                              
                                bias = True,

        )        
        _ = net.load_state_dict(state_dict, strict=False)
        net.update()
        net.to(device) 
        return net, modello_base


    elif args.name_model == "WACNN":
        if args.train_baseline:
            checkpoint = torch.load(pret_checkpoint , map_location=device)
            net = from_state_dict(models["base"], checkpoint)
            net.to(device)
            net.update()
            return net, net
        else:
            if args.origin_model != "base":
                checkpoint = torch.load(pret_checkpoint , map_location=device)#["state_dict"]
                state_dict = checkpoint["state_dict"]
                args_new = checkpoint["args"]

                net = models["gate"](N = args_new.N,
                                    M =args_new.M,
                                    dim_adapter_attn = args_new.dim_adapter_attn,
                                    stride_attn = args_new.stride_attn,
                                    kernel_size_attn = args_new.kernel_size_attn,
                                    padding_attn = args_new.padding_attn,
                                    type_adapter_attn = args_new.type_adapter_attn,
                                    position_attn = args_new.position_attn,
                                    num_adapter = 3 ,#len(args.considered_classes),
                                    aggregation = args_new.aggregation,
                                    std = args_new.std,
                                    mean = args_new.mean,                              
                                    bias = True, # args_new.bias,
                                    skipped = False
                                        ) 
                
    
                _ = net.load_state_dict(state_dict, strict=True)
                return net, net
            
            
            checkpoint = torch.load(pret_checkpoint , map_location=device)#["state_dict"]
            modello_base = from_state_dict(models[args.origin_model], checkpoint)
            modello_base.update()
            modello_base.to(device) 


            net = models["gate"](N = args.N,
                                M =args.M,
                                dim_adapter_attn = args.dim_adapter_attn,
                                stride_attn = args.stride_attn,
                                kernel_size_attn = args.kernel_size_attn,
                                padding_attn = args.padding_attn,
                                type_adapter_attn = args.type_adapter_attn,
                                position_attn = args.position_attn,
                                num_adapter = num_adapter,
                                aggregation = args.aggregation,
                                std = args.std,
                                mean = args.mean,                              
                                bias = True,
                                skipped = args.skipped
                                    ) 

            print("entro qua parte 1")
            state_dict = modello_base.state_dict()
            #state_dict = net.state_dict()

            state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.1.original_model_weights.weight", stringa = "g_s.1.weight" ): v for k, v in state_dict.items()}
            state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.1.original_model_weights.bias", stringa = "g_s.1.bias" ): v for k, v in state_dict.items()}
            state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.3.original_model_weights.weight", stringa = "g_s.3.weight" ): v for k, v in state_dict.items()}
            state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.3.original_model_weights.bias", stringa = "g_s.3.bias" ): v for k, v in state_dict.items()}
            state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.6.original_model_weights.weight", stringa = "g_s.6.weight" ): v for k, v in state_dict.items()}
            state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.6.original_model_weights.bias", stringa = "g_s.6.bias" ): v for k, v in state_dict.items()}
            state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.8.original_model_weights.weight", stringa = "g_s.8.weight" ): v for k, v in state_dict.items()}
            state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.8.original_model_weights.bias", stringa = "g_s.8.bias" ): v for k, v in state_dict.items()}
                

            if args.pret_checkpoint_gate != "none":
                gate_dict = torch.load(args.pret_checkpoint_gate , map_location=device)["state_dict"]
                for k in list(gate_dict.keys()):
                    if "gate." in k:
                        state_dict[k] = gate_dict[k]
                 

            _ = net.load_state_dict(state_dict, strict=False)
            net.to(device)
            return net, modello_base

