"""
Usage: python estimate_impact.py \
    --base_model NousResearch/Llama-2-7b-hf \
    --data_path data/processed_data//c4_processed_50k \
    --output_file data/cherry_indices/llama2-7b-impact.pt
"""

import sys
import os
import math
import warnings
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk


def main(
    # model/data params
    base_model,
    data_path,
    output_file,
    
    # Hessian params
    Hessian_samples: int = 128,
    batch_size: int = 1,
    data_seed: int = 42,
    
    # cherry params
    cherry_fraction: float = 1 / 256,
):
    if batch_size != 1:
        warnings.warn("It is strongly recommended that `batch_size` should be set to 1 for more accurate estimation.")
    
    data = load_from_disk(data_path).shuffle(seed=data_seed).select(range(Hessian_samples))
    print(len(load_from_disk(data_path)))
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, padding_side='right', use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, padding=True, return_tensors='pt'
    )
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map='balanced_low_0', trust_remote_code=True
    )
    model.eval()
    model.config.use_cache = False
    
    num_batches = 0
    H = {}
    for batch in tqdm(dataloader, desc="Estimating Hessian"):
        seq_len = batch[model.main_input_name].shape[1]
        with torch.cuda.amp.autocast():
            loss = model(**batch).loss
            loss *= seq_len
            
        loss.backward()
        
        suffix = '.weight'
        for n, p in model.named_parameters():
            if n.endswith(suffix):
                grad = p.grad
                if grad.ndimension() < 2:
                    continue
                
                n = n[:-len(suffix)]
                if n not in H:
                    H[n] = torch.zeros(*grad.shape, dtype=grad.dtype).cuda()
                else:
                    H[n] *= num_batches / (num_batches + 1)
                
                h = grad ** 2
                H[n] += (h / (num_batches + 1)).cuda()
        
            p.grad.zero_()
        
        num_batches += 1
    
    for k, v in H.items():
        num_cherries = math.ceil(v.shape[-1] * cherry_fraction) // 8 * 8
        print(num_cherries)
        H[k] = v.argsort(descending=True, dim=-1)[:, :num_cherries].to(dtype=torch.int32, device='cpu')
    
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(H, output_file)
    

if __name__ == '__main__':
    import fire
    
    fire.Fire(main)
    