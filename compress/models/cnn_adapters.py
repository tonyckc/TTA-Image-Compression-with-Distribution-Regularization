
from .cnn import WACNN
import torch.nn as nn
from compress.ops import ste_round
from compressai.models.utils import  update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.layers.layers import      Win_noShift_Attention , Win_noShift_Attention_Multiple_Adapter , ResidualMultipleAdaptersDeconv
from compress.layers.utils_function import deconv
from compress.layers.gdn import GDN
import torch 
from compressai.models import CompressionModel
from compress.layers.gate import GateNetwork

import torch.nn.functional as F


# init

class WACNNGateAdaptive(WACNN):

    def __init__(
        self,
        N=192,
        M=320,
        num_adapter = 3,


        dim_adapter_attn: list = [0,0,0],
        stride_attn: list = [1,1,1],
        kernel_size_attn:list = [1,1,1],
        padding_attn: list = [0,0,0],
        position_attn: list = ["res","res","res"],
        type_adapter_attn: list = ["singular","singular","singular"],
        aggregation: str = "top1",
        mid_dim: int = 64,
        bias: bool = True,
        std: float = 0.00,
        mean: float = 0.00,
        groups: int = 1,
        skipped: bool = False,
        all_adapters = False,
        **kwargs
    ):
        super().__init__(N, M, **kwargs)


        self.gate = GateNetwork(in_dim= 320, mid_dim = mid_dim ,num_adapter=num_adapter)
        self.all_adapters = all_adapters 

        if self.all_adapters:
            self.g_s = nn.Sequential(
                Win_noShift_Attention( dim=M, num_heads=8,window_size=4,shift_size=2),  # for the attention (no change)
                ResidualMultipleAdaptersDeconv(M, N, kernel_size=5, stride=2, num_adapter = num_adapter, skipped = skipped, aggregation= aggregation),  #deconv(M, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                #deconv(N, N, kernel_size=5, stride=2), # rimettere 
                ResidualMultipleAdaptersDeconv(N, N, kernel_size=5, stride=2, num_adapter = num_adapter, skipped = skipped, aggregation= aggregation), 
                GDN(N, inverse=True),
                Win_noShift_Attention_Multiple_Adapter( dim=N,num_heads=8,window_size=8,shift_size=4,
                                            dim_adapter=dim_adapter_attn,
                                            kernel_size = kernel_size_attn,
                                            padding = padding_attn,
                                            stride = stride_attn,
                                            bias = bias,
                                            std = std,
                                            position = position_attn,
                                            type_adapter = type_adapter_attn,
                                            mean = mean, 
                                            groups = groups,
                                            num_adapter = num_adapter,
                                            aggregation = aggregation
                                            ),
                ResidualMultipleAdaptersDeconv(N, N, kernel_size=5, stride=2, num_adapter = num_adapter, skipped = skipped, aggregation= aggregation ), #modificare lo state dictddd
                GDN(N, inverse=True),
                ResidualMultipleAdaptersDeconv(N, 3, kernel_size=5, stride=2, num_adapter = num_adapter, skipped = skipped, aggregation= aggregation), #modificare lo state dict
            )
        else:
            self.g_s = nn.Sequential(
                Win_noShift_Attention( dim=M, num_heads=8,window_size=4,shift_size=2), #0  # for the attention (no change)
                deconv(M, N, kernel_size=5, stride=2), #1
                GDN(N, inverse=True), #2
                deconv(N, N, kernel_size=5, stride=2), # rimettere  
                GDN(N, inverse=True),
                Win_noShift_Attention_Multiple_Adapter( dim=N,num_heads=8,window_size=8,shift_size=4,
                                            dim_adapter=dim_adapter_attn,
                                            kernel_size = kernel_size_attn,
                                            padding = padding_attn,
                                            stride = stride_attn,
                                            bias = bias,
                                            std = std,
                                            position = position_attn,
                                            type_adapter = type_adapter_attn,
                                            mean = mean, 
                                            groups = groups,
                                            num_adapter = num_adapter,
                                            aggregation = aggregation
                                            ),
                ResidualMultipleAdaptersDeconv(N, N, kernel_size=5, stride=2, num_adapter = num_adapter, skipped = skipped, aggregation= aggregation ), #modificare lo state dictddd
                GDN(N, inverse=True),
                ResidualMultipleAdaptersDeconv(N, 3, kernel_size=5, stride=2, num_adapter = num_adapter, skipped = skipped, aggregation= aggregation), #modificare lo state dict
            )
        self.length_reconstruction_decoder = len(self.g_s)


    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super(CompressionModel, self).load_state_dict(state_dict, strict=strict)



    def handle_gate_parameters(self, re_grad = True):
        for n,p in self.gate.named_parameters():
            p.requires_grad = re_grad





    def handle_adapters_parameters(self, re_grad = True): 
        #for single_adapter in self.adapter_trasforms:
        for n,p in self.g_s.named_parameters(): 
                #if "skipped" not in n:
                if "adapter_" in n or "AttentionAdapter" in n:
                    p.requires_grad = re_grad 


    

    def forward(self, x, oracle = None):
        y = self.g_a(x)
        # qua metterei il Gate 
        gate_values = self.gate(y) #questi sono i valori su cui fare la softmax 
        if x.shape[0]> 0:
            gate_probs = F.softmax(gate_values, dim = 1)
        else:
            gate_probs = F.softmax(gate_values, dim = 0)

        y_shape = y.shape[2:]
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z, training = False)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []



        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            #_, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu ##ffff
            #y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            #y_hat_slice = y_q_slice + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        if self.all_adapters:
            for j,module in enumerate(self.g_s):
                if j in (0,2,4,7):
                    
                    y_hat = module(y_hat)
                elif j in (1,3,5,6):
                    y_hat = module(y_hat, gate_probs, oracle)
                else: # caso finale in cui j == self.length_reconstruction_decoder -1
                    x_hat = module(y_hat, gate_probs,oracle)
        else:
            for j,module in enumerate(self.g_s):
                if j in (0,1,2,3,4,7):
                    
                    y_hat = module(y_hat)
                elif j in (5,6):
                    y_hat = module(y_hat, gate_probs, oracle)
                else: # caso finale in cui j == self.length_reconstruction_decoder -1
                    x_hat = module(y_hat, gate_probs,oracle)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "logits": gate_values, 
            "y_hat":y_hat
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        
        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])




        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        gate_values = self.gate(y)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


                
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],"gate_values":gate_values}



    def decompress(self, strings, shape, gate_values, oracle = None):


        gate_probs = F.softmax(gate_values)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)



            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

            
        y_hat = torch.cat(y_hat_slices, dim=1)
        if self.all_adapters:
            for j,module in enumerate(self.g_s):
                if j in (0,2,4,7):
                    
                    y_hat = module(y_hat)
                elif j in (1,3,5,6):
                    y_hat = module(y_hat, gate_probs, oracle)
                else: # caso finale in cui j == self.length_reconstruction_decoder -1
                    x_hat = module(y_hat, gate_probs,oracle)
        else:
            for j,module in enumerate(self.g_s):
                if j in (0,1,2,3,4,7):
                    
                    y_hat = module(y_hat)
                elif j in (5,6):
                    y_hat = module(y_hat, gate_probs, oracle)
                else: # caso finale in cui j == self.length_reconstruction_decoder -1
                    x_hat = module(y_hat, gate_probs,oracle)
        return {"x_hat": x_hat, "y_hat":y_hat, "logits":gate_probs}
    


    def forward_gate(self,x):
        y = self.g_a(x)
        # qua metterei il Gate 
        logits = self.gate(y) #questi sono i valori su cui fare la softmax ##
        return {"x_hat": x, "logits": logits}    