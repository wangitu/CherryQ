"""
Usage: python evaluate_perplexity.py \
    NousResearch/Llama-2-7b-hf \
    <directory where the model checkpoints are written>
"""

import sys
import os
from tqdm import trange

import torch
import torch.nn as nn


def get_wikitext2(seqlen, model):
    from datasets import load_dataset
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side='right', use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt').input_ids[0]
    if 'gemma' in model:
        firstenc, subenc = testenc[:seqlen], testenc[seqlen:]
        encs = subenc.split(seqlen - 1)
        encs = [torch.cat([torch.tensor([tokenizer.bos_token_id]), enc]) for enc in encs]
        testenc = torch.cat([firstenc, *encs])

    return testenc


def get_c4(seqlen, model):
    from datasets import load_dataset
    valdata = load_dataset(
        'allenai/c4', 'en', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side='right', use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    
    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if 'gemma' in model:
            input_ids = tmp.input_ids[0, :seqlen] # gemma must have bos token as its first token, otherwise extrodinarily high perplexity
        else:
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            input_ids = tmp.input_ids[0, i:j]
        valenc.append(input_ids)
    valenc = torch.cat(valenc, dim=0)

    return valenc


def get_loaders(
    name, seqlen=2048, model=''
):
    if 'c4' in name:
        return get_c4(seqlen, model)
    elif 'wikitext2' in name:
        return get_wikitext2(seqlen, model)
    else:
        raise ValueError(f"Unsupported dataset {name}.")


@torch.inference_mode
def eval_ppl(model, testenc, seqlen=2048):
    print('Evaluating ...')
    nsamples = testenc.numel() // seqlen
    print(nsamples)
    testenc = testenc.cuda()
    nlls = []
    for i in trange(nsamples):
        batch = testenc[i * seqlen: (i + 1) * seqlen].unsqueeze(0).cuda()
        lm_logits = model(input_ids=batch).logits
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl.item())
    

def main(
    tokenizer_path: str,
    quant_model_path: str
):
    from cherryq.utils import from_quantized
    model = from_quantized(quant_model_path, torch_dtype=torch.float16, device_map={'': 0})
    model.eval()
    model.config.use_cache = False

    from cherryq.nn_modules.qlinear import QuantLinear
    
    with QuantLinear.dmode(layerwise_dequantize=False):
        eval_ppl(model, get_loaders('c4', model=tokenizer_path))
        eval_ppl(model, get_loaders('wikitext2', model=tokenizer_path))
    
    
if __name__ == '__main__':
    import fire
    
    fire.Fire(main)
    