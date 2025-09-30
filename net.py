import torch
from compressai.models.google import ScaleHyperprior
from compressai.models.google import ScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional, EntropyModel
from compressai.models import JointAutoregressiveHierarchicalPriors,MeanScaleHyperprior
from sga import Quantizator_SGA
import numpy as np
import math
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
#from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model

from compressai.models.base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from compressai.models.utils import conv, deconv
import torch.nn as nn

# modified from https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/main/sga.py#L110-L121
# Copyright (c) 2020 mandt-lab Licensed under The MIT License
def quantize_sga(y: torch.Tensor, tau: float, medians=None, eps: float = 1e-5):
    # use Gumbel Softmax implemented in tfp.distributions.RelaxedOneHotCategorical

    # (N, C, H, W)
    if medians is not None:
        y -= medians
    y_floor = torch.floor(y)
    y_ceil = torch.ceil(y)
    # (N, C, H, W, 2)
    y_bds = torch.stack([y_floor, y_ceil], dim=-1)
    # (N, C, H, W, 2)
    ry_logits = torch.stack(
        [
            -torch.atanh(torch.clamp(y - y_floor, -1 + eps, 1 - eps)) / tau,
            -torch.atanh(torch.clamp(y_ceil - y, -1 + eps, 1 - eps)) / tau,
        ],
        axis=-1,
    )
    # last dim are logits for DOWN or UP
    ry_dist = torch.distributions.RelaxedOneHotCategorical(tau, logits=ry_logits)
    ry_sample = ry_dist.rsample()
    outputs = torch.sum(y_bds * ry_sample, dim=-1)
    if medians is not None:
        outputs += medians
    return outputs

class EntropyBottleneckNoQuant(EntropyBottleneck):
    def __init__(self, channels):
        super().__init__(channels)
        self.sga = Quantizator_SGA()

    def forward(self, x_quant):
        #print(x_quant.shape)
        perm = np.arange(len(x_quant.shape))
        perm[0], perm[1] = perm[1], perm[0]
        # Compute inverse permutation
        inv_perm = np.arange(len(x_quant.shape))[np.argsort(perm)]
        x_quant = x_quant.permute(*perm).contiguous()
        shape = x_quant.size()
        x_quant = x_quant.reshape(x_quant.size(0), 1, -1)

        likelihood,_ ,_ = self._likelihood(x_quant)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        # Convert back to input tensor shape
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return likelihood

class GaussianConditionalNoQuant(GaussianConditional):
    def __init__(self, scale_table):
        super().__init__(scale_table=scale_table)

    def forward(self, x_quant, scales, means):
        likelihood = self._likelihood(x_quant, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return likelihood

class ScaleHyperpriorSGA(ScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneckNoQuant(N)
        self.gaussian_conditional = GaussianConditionalNoQuant(None)
        self.sga = Quantizator_SGA()

    def quantize(self, inputs, mode, means=None, it=None, tot_it=None,tau=None):
        if means is not None:
            inputs = inputs - means
        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            outputs = inputs + noise
        elif mode == "round":
            outputs = torch.round(inputs)
        elif mode == "sga":
            outputs = quantize_sga(inputs,tau)#self.sga(inputs, it, "training", tot_it)
        else:
            assert(0)
        if means is not None:
            outputs = outputs + means
        return outputs

    def dropout_h_a(self, data, p=0.5, dropout=False, dropN=1):
        if dropN == 1:
            model_sub = nn.Sequential(*list(self.h_a.children())[:-1])
            output = model_sub(data)
            if dropout:
                output = F.dropout(output, p=p, training=dropout)
            model_sub = nn.Sequential(list(self.h_a.children())[-1])
            output = model_sub(output)
            return output

    def variance(self,batch_data):
            mean = torch.mean(batch_data, dim=0)  # Compute mean along the batch dimension
            squared_diff = torch.square(batch_data - mean)  # Calculate squared differences
            variance = torch.mean(squared_diff, dim=0)  # Compute mean along the batch dimension
            variance = variance.unsqueeze(0)  # Add a dimension of size 1 at the beginning
            return variance

    def forward(self, x, mode, y_in=None, z_in=None, it=None, tot_it=None,tau=None,dropN=1):
        if mode == "init":
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
        else:
            y = y_in
            z = z_in
        if mode == "init" or mode == "round":
            y_hat = self.quantize(y, "round")
            z_hat = self.quantize(z, "round")
        elif mode == "noise":
            y_hat = self.quantize(y, "noise")
            z_hat = self.quantize(z, "noise")
        elif mode =="sga":

            y_hat = self.quantize(y, "sga", None, it, tot_it, tau)
            z_hat = self.quantize(z, "sga", None, it, tot_it, tau)
        else:
            assert(0)
        if mode == 'sga':
            z_likelihoods = self.entropy_bottleneck(z_hat)
        else:
            z_likelihoods = self.entropy_bottleneck(z_hat)
        scales_hat = self.h_s(z_hat)

        log2pi = torch.log(torch.tensor(2. * np.pi)).cuda()

        T_sample = 20
        repeat_data = y.repeat(T_sample, 1, 1, 1)
        output = self.dropout_h_a(repeat_data, dropout=True,dropN=dropN)
        mean_out = torch.mean(output, dim=0, keepdim=True)
        variance_out = self.variance(output)
        log_pdf = -.5 * (((z - mean_out) ** 2. / variance_out) + torch.log(variance_out) + log2pi)
        mse = (z - mean_out) ** 2.
        if y_hat.shape != scales_hat.shape:
            scales_hat = scales_hat[:,:,:min(y_hat.shape[2],scales_hat.shape[2]),:min(y_hat.shape[3],scales_hat.shape[3])]
        y_likelihoods = self.gaussian_conditional(y_hat, scales_hat, None)
        x_hat = self.g_s(y_hat)
        return {
            "y": y.detach().clone(),
            "z": z.detach().clone(), 
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods,'bpp':log_pdf},
            "mse":mse
        }


class h_a(nn.Module):
    def __init__(self, N):
            super(h_a, self).__init__()
            self.l1 = conv3x3(N, N)
            self.lr1 = nn.LeakyReLU(inplace=True)
            self.l2 = conv3x3(N, N)
            self.lr2 = nn.LeakyReLU(inplace=True)
            self.l3 = conv3x3(N, N, stride=2)
            self.lr3 = nn.LeakyReLU(inplace=True)
            self.l4 = conv3x3(N, N)
            self.lr4 = nn.LeakyReLU(inplace=True)
            self.l5 = conv3x3(N, N, stride=2)
    def forward(self,data, dropout=False):
            x = self.l1(data)
            x = self.lr1(x)
            x = self.l2(x)
            x = self.lr2(x)
            x = self.l3(x)
            x = self.lr3(x)
            x = self.l4(x)
            x = self.lr4(x)
            x = F.dropout(x,p=0.5,training=dropout)
            x = self.l5(x)
            return x

class Cheng2020AnchorSGA(JointAutoregressiveHierarchicalPriors):

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.entropy_bottleneck = EntropyBottleneckNoQuant(N)
        self.gaussian_conditional  = GaussianConditionalNoQuant(None)
        self.sga = Quantizator_SGA()
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def dropout_h_a(self, data, p=0.5, dropout=False, dropN=1):
      if dropN == 1:
        model_sub = nn.Sequential(*list(self.h_a.children())[:-1])
        output = model_sub(data)
        if dropout:
            output = F.dropout(output, p=p, training=dropout)
        model_sub = nn.Sequential(list(self.h_a.children())[-1])
        output = model_sub(output)
        return output
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net

    def quantize(self, inputs, mode, means=None, it=None, tot_it=None,tau=None):
        if means is not None:
            inputs = inputs - means
        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            outputs = inputs + noise
        elif mode == "round":
            outputs = torch.round(inputs)
        elif mode == "sga":
            outputs = quantize_sga(inputs,tau) #self.sga(inputs, it, "training", tot_it)
        else:
            assert(0)
        if means is not None:
            outputs = outputs + means
        return outputs

    def variance(self,batch_data):
            mean = torch.mean(batch_data, dim=0)  # Compute mean along the batch dimension
            squared_diff = torch.square(batch_data - mean)  # Calculate squared differences
            variance = torch.mean(squared_diff, dim=0)  # Compute mean along the batch dimension
            variance = variance.unsqueeze(0)  # Add a dimension of size 1 at the beginning
            return variance
    def forward(self, x, mode, y_in=None, z_in=None, it=None, tot_it=None,tau=None,dropN=1):

        if mode == "init":

            y = self.g_a(x)
            z = self.h_a(y)
        else:
            y = y_in
            z = z_in

        if mode == "init" or mode == "round":
            y_hat = self.quantize(y, "round")
            z_hat = self.quantize(z, "round")
        elif mode == "noise":
            y_hat = self.quantize(y, "noise")
            z_hat = self.quantize(z, "noise")
        elif mode =="sga":
            y_hat = self.quantize(y, "sga", None, it, tot_it,tau)
            z_hat = self.quantize(z, "sga", None, it, tot_it,tau)
        else:
            assert(0)


        z_likelihoods = self.entropy_bottleneck(z_hat)
        params = self.h_s(z_hat)


        if y_hat.shape != params.shape:
            params = params[:,:,:min(y_hat.shape[2],params.shape[2]),:min(y_hat.shape[3],params.shape[3])]

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        log2pi = torch.log(torch.tensor(2. * np.pi)).cuda()


        p = 0.5 # dropout probability
        T_sample = 20 # number of MC sampling
        repeat_data = y.repeat(T_sample,1,1,1)
        output = self.dropout_h_a(repeat_data, p=p,dropout=True,dropN=dropN)
        mean_out = torch.mean(output,dim=0,keepdim=True)
        # avoid too small sigma for stable training
        sigma_threshold = 1e-4

        variance_out = torch.clamp(self.variance(output), min=sigma_threshold ** 2)
        log_pdf = -.5 * ((((z - mean_out) ** 2.) / variance_out) + torch.log(variance_out) + log2pi)

        mse = (z - mean_out) ** 2.


        y_likelihoods = self.gaussian_conditional(y_hat, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {
            "y": y.detach().clone(),
            "z": z.detach().clone(),
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods,'bpp':log_pdf},
            'means': means_hat,
            'scales': scales_hat,
            'mse':mse
        }



class Cheng2020Anchor(JointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
