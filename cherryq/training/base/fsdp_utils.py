import sys
import os
import warnings

from torch.distributed.fsdp import _state_dict_utils
from torch.distributed.fsdp._common_utils import clean_tensor_name
    
    
DefaultFullPostStateDictHook = _state_dict_utils._full_post_state_dict_hook
    
    
def low_gpu_full_post_state_dict_hook(module, fsdp_state, state_dict, prefix):
    
    def param_hook(state_dict, prefix, fqn):
        clean_key = fqn
        clean_prefix = clean_tensor_name(prefix)
        # Strip prefix out of key if needed as buffer names and param names
        # do not have prefix considered as they are not computed in `state_dict`
        # call.
        if clean_key.startswith(clean_prefix):
            clean_key = clean_key[len(clean_prefix) :]

        # Clone parameters before exiting the `_unshard_fsdp_state_params()` context.
        if not getattr(state_dict[fqn], "_has_been_cloned", False):
            try:
                state_dict[fqn] = state_dict[fqn].cpu().clone().detach()
                state_dict[fqn]._has_been_cloned = True  # type: ignore[attr-defined]
            except BaseException as e:
                warnings.warn(
                    f"Failed to clone() tensor with name {fqn} on rank {fsdp_state.rank}. "
                    "This may mean that this state_dict entry could point to invalid "
                    "memory regions after returning from state_dict() call if this "
                    "parameter is managed by FSDP. Please check clone "
                    f"implementation of {fqn}. Error: {str(e)}"
                )

    return _state_dict_utils._common_unshard_post_state_dict_hook(
        module, fsdp_state, state_dict, prefix, param_hook
    )
    

# enable to efficiently saving `state_dict` for fsdp
def enable_low_gpu_full_post_state_dict_hook():
    _state_dict_utils._full_post_state_dict_hook = low_gpu_full_post_state_dict_hook
    
def disable_low_gpu_full_post_state_dict_hook():
    _state_dict_utils._full_post_state_dict_hook = DefaultFullPostStateDictHook
    