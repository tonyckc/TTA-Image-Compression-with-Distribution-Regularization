from compress.layers.base import ResidualBlock, ResidualBlockUpsample,  AttentionBlock

from compressai.models.utils import  update_registered_buffers
from compressai.models import Cheng2020Attention
from compressai.models import CompressionModel
from torch import nn
from compress.layers.gate import GateNetwork
from compress.layers.layers import AttentionBlockWithMultipleAdapters, ResidualBlockUpsampleMultipleAdapters, ResidualBlockMultipleAdapters, subpel_conv3x3MultipleAdapters
import torch
import torch.nn.functional as F
import warnings
from compressai.ans import BufferedRansEncoder, RansDecoder

class Cheng2020AttnAdapter(Cheng2020Attention):
    def __init__(
        self,
        N=192,
        num_adapter = 3,
        dim_adapter_attn: list = [0,0,0],
        stride_attn: list = [1,1,1],
        kernel_size_attn:list = [1,1,1],
        padding_attn: list = [0,0,0],
        position_attn: list = ["res","res","res"],
        type_adapter_attn: list = ["singular","singular","singular"],
        aggregation: str = "top1",
        bias: bool = True,
        std: float = 0.00,
        mean: float = 0.00,
        groups: int = 1,
        **kwargs
    ):
        super().__init__(N, **kwargs)


        self.gate = GateNetwork(in_dim= N, mid_dim = int(N//3),num_adapter=num_adapter)

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlockWithMultipleAdapters(N,
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
            ResidualBlockMultipleAdapters(N,
                                          N,  
                                          num_adapter = num_adapter,  
                                          aggregation= aggregation),
            ResidualBlockUpsampleMultipleAdapters(N, N, 2,
                                          num_adapter = num_adapter,  
                                          aggregation= aggregation),
            ResidualBlockMultipleAdapters(N,
                                          N,  
                                          num_adapter = num_adapter,  
                                          aggregation= aggregation),
            subpel_conv3x3MultipleAdapters(N, 3, 2),
        )



    def freeze_net(self):
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        for p in self.parameters(): 
            p.requires_grad = False



    def handle_gate_parameters(self, re_grad = True):
        for n,p in self.gate.named_parameters():
            p.requires_grad = re_grad





    def handle_adapters_parameters(self, re_grad = True): 
        #for single_adapter in self.adapter_trasforms:
        for n,p in self.g_s.named_parameters(): 
                #if "skipped" not in n:
                if "adapter_" in n or "AttentionAdapter" in n:
                    p.requires_grad = re_grad 



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


    
    def forward(self, x,  oracle = None):
        y = self.g_a(x)


        gate_values = self.gate(y) #questi sono i valori su cui fare la softmax 
        if x.shape[0]> 0:
            gate_probs = F.softmax(gate_values, dim = 1)
        else:
            gate_probs = F.softmax(gate_values, dim = 0)


        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)






        for j,module in enumerate(self.g_s):
            if j in (0,1,2,3,4):
                y_hat = module(y_hat)
            elif j in (5,6,7,8):
                y_hat = module(y_hat, gate_probs.to("cuda"), oracle)
            else: # caso finale in cui j == self.length_reconstruction_decoder -1
                x_hat = module(y_hat, gate_probs.to("cuda"),oracle)



        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "logits": gate_values, 
            "y_hat":y_hat
        }
    

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a(x)
        z = self.h_a(y)

        gate_values = self.gate(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "gate_values":gate_values}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape,gate_values, oracle = None):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        gate_probs = F.softmax(gate_values)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))

        for j,module in enumerate(self.g_s):
            if j in (0,1,2,3,4):
                y_hat = module(y_hat)
            elif j in (5,6,7,8):
                y_hat = module(y_hat, gate_probs.to("cuda"), oracle)
            else: # caso finale in cui j == self.length_reconstruction_decoder -1
                x_hat = module(y_hat, gate_probs.to("cuda"),oracle).clamp_(0, 1)



 
        return {"x_hat": x_hat, "y_hat":y_hat, "logits":gate_probs}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp:wp+1]=rv