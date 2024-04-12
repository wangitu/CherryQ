from typing import Optional
from collections import namedtuple

import torch


iinfo = namedtuple("iinfo", ["min", "max"])


def get_max_input(input, clip_val, layerwise):
    if clip_val is not None:
        input = torch.clamp(input, clip_val[0], clip_val[1])
    # NOTE: dynamic scaling (max_input).
    if layerwise:
        max_input = torch.max(torch.abs(input))
    else:
        if input.ndimension() <= 3: # (out, in) / (bs, seq, hid)
            # weight & hidden layer
            max_input = (
                torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                .detach()
            )
        elif input.ndimension() == 4:
            # TODO: attention score matrix, calculate alpha / beta per head
            tmp = input.view(input.shape[0], input.shape[1], -1)
            max_input = (
                torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                .unsqueeze(-1)
                .detach()
            )
        else:
            raise ValueError(f"Not implemented for shape: {input.shape}")
        
    return max_input


class Quantizer:
    @staticmethod
    def iinfo(num_bits) -> iinfo:
        ...
        
    @staticmethod
    def get_scaling_factor(input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: scaling factor
        """

    @staticmethod
    def quantize(input, S, num_bits):
        ...
    
    @staticmethod
    def dequantize(rounded_input, S):
        ...
        
    @staticmethod
    def transform(input, S, num_bits):
        ...
    

class RestrictedSymQuantizer(Quantizer, torch.autograd.Function):
    """
    uniform quantization with restricted range [-(2 ** (num_bits - 1) - 1), 2 ** (num_bits - 1) - 1]
    """
    
    @staticmethod
    def iinfo(num_bits):
        return iinfo(
            min=-(2 ** (num_bits - 1) - 1),
            max=2 ** (num_bits - 1) - 1
        )
    
    @staticmethod
    def get_scaling_factor(input, clip_val, num_bits, layerwise):
        max_input = get_max_input(input, clip_val, layerwise)
        s = (max_input + 1e-6) / (2 ** (num_bits - 1) - 1)
        return s
    
    @staticmethod
    def quantize(input, S, num_bits):
        s = S.expand_as(input)
        q_range = [-(2 ** (num_bits - 1) - 1), 2 ** (num_bits - 1) - 1]
        return torch.clamp(torch.round((input / s).float()), q_range[0], q_range[1]).to(torch.int32)
    
    @staticmethod
    def dequantize(rounded_input, S):
        s = S.expand_as(rounded_input)
        return rounded_input.to(s.dtype) * s
    
    @staticmethod
    def transform(input, S, num_bits):
        s = S.expand_as(input)
        q_range = [-(2 ** (num_bits - 1) - 1), 2 ** (num_bits - 1) - 1]
        rounded_input = torch.clamp(torch.round(input / s), q_range[0], q_range[1])
        return rounded_input * s

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, S: Optional[torch.Tensor]=None):
        """
        Backward compatibility for naive QAT
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param num_bits: number of bits
        :param S: overriding scaling factor
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.clamp(input, clip_val[0], clip_val[1]) if clip_val is not None else input
        
        if S is None:
            S = RestrictedSymQuantizer.get_scaling_factor(input, clip_val, num_bits, layerwise)
        s = S.expand_as(input)
        
        output = torch.round(input / s) * s
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward compatibility for naive QAT
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        if clip_val is not None:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None
    

class FullSymQuantizer(Quantizer, torch.autograd.Function):
    """
    uniform quantization with full range [-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1]
    """
     
    @staticmethod
    def iinfo(num_bits):
        return iinfo(
            min=-(2 ** (num_bits - 1)),
            max=2 ** (num_bits - 1) - 1
        )
    
    @staticmethod
    def get_scaling_factor(input, clip_val, num_bits, layerwise):
        max_input = get_max_input(input, clip_val, layerwise)
        s = (max_input + 1e-6) / (2 ** (num_bits - 1))
        return s
    
    @staticmethod
    def quantize(input, S, num_bits):
        s = S.expand_as(input)
        qmax = 2 ** (num_bits - 1) - 1e-2
        return torch.round((torch.clamp((input / s).float(), -qmax, qmax)) - 0.5).to(torch.int32)
    
    @staticmethod
    def dequantize(rounded_input, S):
        s = S.expand_as(rounded_input)
        return (rounded_input.to(s.dtype) + 0.5) * s
    
    @staticmethod
    def transform(input, S, num_bits):
        s = S.expand_as(input)
        qmax = 2 ** (num_bits - 1) - 1e-2
        rounded_input = torch.round(torch.clamp(input / s, -qmax, qmax) - 0.5)
        return (rounded_input + 0.5) * s
    
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, S: Optional[torch.Tensor]=None):
        """
        Backward compatibility for naive QAT
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param num_bits: number of bits
        :param S: overriding scaling factor
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.clamp(input, clip_val[0], clip_val[1]) if clip_val is not None else input
        
        if S is None:
            S = FullSymQuantizer.get_scaling_factor(input, clip_val, num_bits, layerwise)
        s = S.expand_as(input)
        
        qmax = 2 ** (num_bits - 1) - 1e-2
        rounded_input = torch.round(torch.clamp(input / s, -qmax, qmax) - 0.5)
        output = (rounded_input + 0.5) * s
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward compatibility for naive QAT
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        if clip_val is not None:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None
    