"""
Usage: python preprocess.py \
    --model-path lmsys/vicuna-7b-v1.5 \
    --model-id vicuna_v1.1 \
    --in-file raw_data/chat \
    --out-dir processed_data/sharegpt_processed_20k
"""

import sys
import os
import json
import argparse

from datasets import Dataset
from transformers import AutoTokenizer
from conversation import get_conv_template


def load_data(in_file, max_recursion_depth=1):
    if in_file.endswith('json'):
        with open(in_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif in_file.endswith('jsonl'):
        with open(in_file, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                data.append(json.loads(line.strip()))
    elif os.path.isdir(in_file) and max_recursion_depth > 0:
        data = []
        for file_or_folder in os.listdir(in_file):
            data.extend(
                load_data(os.path.join(in_file, file_or_folder), max_recursion_depth - 1)
            )
    else:
        raise ValueError(f"Loading script for {in_file} is not implemented.")
    
    return data


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, revision=args.revision, use_fast=False, trust_remote_code=True # Whether or not to allow for custom models defined on the Hub executing code on your local machine
    )
    
    def tokenize(data_point):
        if args.model_id is None: # base
            prompt = data_point['text']
        else: # chat
            conv = get_conv_template(args.model_id)
            turns= data_point["conversations"]
            for turn in turns:
                if turn['from'] == 'human':
                    conv.append_message(conv.roles[0], turn['value'])
                elif turn['from'] == 'gpt':
                    conv.append_message(conv.roles[1], turn['value'])
                else:
                    raise ValueError(f"Unrecognized role: {turn['from']}.")
            prompt = conv.get_prompt()
    
        result = tokenizer(
            prompt, padding=False, truncation=True, max_length=args.max_length, return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    data = load_data(args.in_file)
    if args.model_id is not None: # chat
        data = [{"conversations": d["items"]} for d in data]
        
    data = Dataset.from_list(data)
    data = data.map(tokenize, num_proc=8, remove_columns=list(data.features))
    if args.model_id is None:
        data = data.filter(lambda x: len(x['input_ids']) >= 2048, num_proc=8)
    print(data)
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    data.save_to_disk(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, help="A custom name for the model."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2048)

    args = parser.parse_args()
    
    main(args)
    os.path.splitext