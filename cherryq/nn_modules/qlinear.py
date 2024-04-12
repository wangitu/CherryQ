import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from ..core.quantizer import Quantizer
from ..core.cherry_linear import QuantLinear as QuantLinearForTraining


class QuantLinear(nn.Module):
    
    _layerwise_dequantize = True
    
    @classmethod
    @contextmanager
    def dmode(cls, layerwise_dequantize=True):
        mode = cls._layerwise_dequantize
        cls._layerwise_dequantize = layerwise_dequantize
        yield
        cls._layerwise_dequantize = mode
    
    def __init__(
        self, w_bits, group_size, quantizer: Quantizer, cherry_fraction: float,
        in_features, out_features, bias=True, **kwargs
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bits = w_bits
        self.group_size = group_size if group_size != -1 else in_features
        self.quantizer = quantizer
        self.cherry_fraction = cherry_fraction
        self.num_cherries = math.ceil(in_features * cherry_fraction) // 8 * 8
        self.num_normal = self.in_features - self.num_cherries
        
        assert in_features % 8 == 0
        assert out_features % 8 == 0
        
        self.register_buffer(
            "cherry_indices",
            torch.zeros((self.out_features, self.num_cherries), dtype=torch.int16)
        )
        self.register_buffer(
            "cherry_weight",
            torch.zeros((self.out_features, self.num_cherries), dtype=torch.float16)
        )
        self.register_buffer(
            "qweight",
            torch.zeros((out_features, self.num_normal // 8 * self.w_bits), dtype=torch.int8)
        )
        self.register_buffer(
            "scales",
            torch.zeros((out_features, math.ceil(self.num_normal / self.group_size)), dtype=torch.float16)
        )
        
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16))
        else:
            self.bias = None
        
        self.dequantized = False
            
    @torch.inference_mode()
    def pack(self, linear: QuantLinearForTraining):
        if linear.bias is not None:
            self.bias = linear.bias.half().cpu()
        
        cherry_indices = linear.cherry_indices.long()
        self.cherry_indices = (linear.cherry_indices + torch.iinfo(torch.int16).min).to(dtype=torch.int16).cpu() # (num_cherries, out)
        self.cherry_weight = linear.weight.gather(1, cherry_indices).to(torch.float16).cpu()
        normal_indices = torch.ones_like(linear.weight, dtype=torch.bool).scatter(1, cherry_indices, 0)
        normal_weight = linear.weight[normal_indices].reshape(linear.weight.shape[0], -1) # (out, num_normal)
        scales = linear._get_scaling_factors(normal_weight) # (out, num_group)
        self.scales = scales.half().cpu()
        
        scales = scales.repeat_interleave(repeats=linear.group_size, dim=1)[:, :normal_weight.shape[1]] # (out, num_normal)
        intweight = linear.quantizer.quantize(normal_weight, scales, linear.w_bits)
        intweight = intweight.t().contiguous().cpu().numpy() # (num_normal, out)
        intweight = (intweight - linear.quantizer.iinfo(linear.w_bits).min).astype(np.uint8) # The sign bit will be swallowed by bitwise operation if np.int8
        
        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 8 * self.w_bits, intweight.shape[1]), dtype=np.uint8) # (num_normal // 8 * w_bits, out)
        if self.w_bits in [2, 4, 8]:
            while row < qweight.shape[0]:
                for j in range(i, i + (8 // self.w_bits)):
                    qweight[row] |= intweight[j] << (self.w_bits * (j - i))
                i += 8 // self.w_bits
                row += 1
        elif self.w_bits == 3:
            while row < qweight.shape[0]:
                for j in range(i, i + 2):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 2
                qweight[row] |= intweight[i] << 6
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i,  i + 2):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 2
                qweight[row] |= intweight[i] << 7
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 2):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 2
                row += 1
        else:
            raise NotImplementedError(f"{self.w_bits} bit is not implemented yet.")
        
        self.qweight = torch.from_numpy(qweight.astype(np.int8).T) # (out, num_normal // 8 * w_bits)
    
    @torch.inference_mode()
    def unpack(self):
        if self.w_bits in [2, 4, 8]:
            wf = torch.arange(0, 8, self.w_bits)[None, None, :].to(self.qweight)
            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 2).expand(-1, -1, 8 // self.w_bits), # (out, in // 8 * self.w_bits, 8 // self.w_bits)
                wf # (1, 1, 8 // self.w_bits)
            ).to(torch.uint8)
            weight = torch.bitwise_and(weight, (2 ** self.w_bits) - 1)
        elif self.w_bits == 3:
            weight = self.qweight.reshape(
                self.qweight.shape[0], self.qweight.shape[1] // 3, 3, 1
            ).expand(-1, -1, -1, 4) # (out, in // 8, 3, 4)
            wf = torch.tensor(
                [
                    [0, 3, 6, 0],
                    [0, 1, 4, 7],
                    [0, 2, 5, 0]
                ],
            ).reshape(1, 3, 4).to(self.qweight)
            weight = (weight >> wf.unsqueeze(0)) & 0x7
            weight[:, :, 0, 2] = (weight[:, :, 0, 2] & 0x3) | ((weight[:, :, 1, 0] << 2) & 0x4)
            weight[:, :, 1, 3] = (weight[:, :, 1, 3] & 0x1) | ((weight[:, :, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, :, 0, :3], weight[:, :, 1, 1:4], weight[:, :, 2, 1:3]], dim=2) # (out, in // 8, 8)
        else:
            raise NotImplementedError(f"{self.w_bits} bit is not implemented yet.")
        
        weight = weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2]) # (out, in)
        weight += self.quantizer.iinfo(self.w_bits).min
        s = self.scales.repeat_interleave(repeats=self.group_size, dim=1)[:, :weight.shape[1]] # (out, in)
        qweight = self.quantizer.dequantize(weight, s)
        
        weight = torch.zeros(self.out_features, self.in_features).to(qweight) # (out, in)
        cherry_indices = (self.cherry_indices - torch.iinfo(torch.int16).min).long()
        normal_indices = torch.ones_like(weight, dtype=torch.bool).scatter(1, cherry_indices, 0)
        weight[normal_indices] = qweight.flatten()
        weight.scatter_(1, cherry_indices, self.cherry_weight)
        
        return weight
    
    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        if self._layerwise_dequantize:
            weight = self.unpack()
        else:
            if not self.dequantized:
                self.qweight = self.unpack()
                
                del self.cherry_indices
                del self.cherry_weight
                del self.scales
                torch.cuda.empty_cache()
                
                self.dequantized = True
                
            weight = self.qweight

        out = nn.functional.linear(x, weight)
        if self.bias is not None:
            out += self.bias
        return out
    