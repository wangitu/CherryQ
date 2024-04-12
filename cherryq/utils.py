import os
from typing import Union
from tqdm import tqdm

import torch
import torch.nn as nn
import accelerate
import transformers

from .core.cherry_linear import QuantLinear as QuantLinearForTraining
from .nn_modules.qlinear import QuantLinear


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


# 1. avoid early return if `"" in device_map` to add `AlignDevicesHook` to module
# 2. `io_same_device=False` for efficiency
def simple_dispatch_model(model, device_map):
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook
        
    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(m, execution_device=main_device, prev_module_hook=prev_hook)
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(model, cpu_offload_group[0][0])._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=False, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


def make_quant(module: nn.Module, name=''):    
    linears = {}
    if isinstance(module, QuantLinear):
        return linears
    
    for attr in dir(module):
        sub_module = getattr(module, attr)
        if isinstance(sub_module, QuantLinearForTraining):
            device = get_device(sub_module)
            name1 = name + '.' + attr if name != '' else attr
            linears[name1] = sub_module
            delattr(module, attr)
            new_module = QuantLinear(
                sub_module.w_bits, sub_module.group_size, sub_module.quantizer, sub_module.cherry_fraction,
                sub_module.in_features, sub_module.out_features, bias=sub_module.bias
            )
            setattr(module, attr, new_module.to(device))
            
    for name1, child in module.named_children():
        linears.update(make_quant(child, name + '.' + name1 if name != '' else name1))
    
    return linears
        

def pack_model(
    model: nn.Module,
    force_to_cpu=False
):
    if force_to_cpu:
        model.cpu()
        
    linears = make_quant(model)
    
    for name, module in tqdm(model.named_modules(), total=len(list(model.modules())), desc='Packing model...'):
        if isinstance(module, QuantLinear):
            module.pack(linears[name])


def save_quantized(model: nn.Module, save_dir):
    model.cpu()
    
    w_bits = None
    group_size = None
    for module in model.modules():
        if isinstance(module, QuantLinear):
            w_bits = module.w_bits
            group_size = module.group_size
            break
    
    model_base_name = f"pytorch_model-{w_bits}bit-{group_size}g"
    model_save_name = model_base_name + '.bin'
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, model_save_name))
    model.config.save_pretrained(save_dir)
    if hasattr(model, 'generation_config'):
        model.generation_config.save_pretrained(save_dir)
        

from .models.llama.configuration_llama_quant import LlamaConfig
from .models.llama.modeling_llama_quant import LlamaForCausalLM

CAUSALLM_MAP = {
    "llama": (LlamaConfig, LlamaForCausalLM)
}


def from_quantized(
    model_name_or_path,
    model_type='llama',
    torch_dtype=None,
    device_map=None
):
    config_cls, model_cls = CAUSALLM_MAP[model_type]
    if torch_dtype is None:
        torch_dtype = torch.float16
    
    init_contexts = [transformers.modeling_utils.no_init_weights(), accelerate.init_empty_weights(include_buffers=False)]
    with transformers.utils.ContextManagers(init_contexts):
        config = config_cls.from_pretrained(model_name_or_path)
        model = model_cls._from_config(config, torch_dtype=torch_dtype)
        make_quant(model)
        model.tie_weights()
        
    model_name = os.path.join(model_name_or_path, f"pytorch_model-{config.w_bits}bit-{config.group_size}g.bin")
    model_name = os.path.normpath(model_name)
    accelerate.utils.modeling.load_checkpoint_in_model(
        model, checkpoint=model_name, device_map=device_map  
    )
    model = simple_dispatch_model(model, device_map)
        
    return model
