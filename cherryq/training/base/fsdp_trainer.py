import sys
import os

import transformers

from .fsdp_utils import enable_low_gpu_full_post_state_dict_hook


class FSDPTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # extention starts here: finally applying `low_gpu_full_post_state_dict_hook` for fsdp `state_dict`
        enable_low_gpu_full_post_state_dict_hook()
        # extention ends here
