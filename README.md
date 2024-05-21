# CherryQ

## <a id="overview"></a>Overview

This work reveals the phenomenon of parameter heterogeneity in large language models (LLMs). The heterogeneity lies in that a small subset of ”cherry” parameters exhibit a disproportionately large impact on model performance, while the vast majority of parameters have minimal impact. By utilizing this heterogeneity that is prevalent across different model families, scales, and types, CherryQ is proposed to identify and preserve the critical cherry parameters in high precision while aggressively quantizing the remaining parameters to low precision. It empirically outperforms existing quantization approaches in terms of perplexity and downstream task performance.

<p align="center" width="100%">
<a><img src="assets/equation5.png" alt="equation5" style="width: 40%; margin: auto"></a>
<a><img src="assets/hscore.png" alt="hscore" style="width: 100%; margin: auto"></a>
</p>

## Installation

### Create conda environment
```Bash
conda create -n cherryq python=3.10 && conda activate cherryq
``` 

### Install dependencies
```Bash
pip install -r requirements.txt
```

## Data preparation

Enter the `data` directory and:

### 1. Download data into `data/raw_data`
Run the following command:
```Bash
bash download_data.sh
```

### 2. Preprocess (format and tokenize data samples)
For base model (we use C4 as our calibration dataset), run the following command:
```Bash
python preprocess.py \
    --model-path NousResearch/Llama-2-7b-hf \
    --in-file raw_data/base \
    --out-dir processed_data/c4_processed_50k
```

For chat model (we use ShareGPT as our calibration dataset), run the following command:
```Bash
python preprocess.py \
    --model-path lmsys/vicuna-7b-v1.5 \
    --model-id vicuna_v1.1 \
    --in-file raw_data/chat \
    --out-dir processed_data/sharegpt_processed_20k
```

## Estimate parameter impact on model performance
To obtain the cherry indices that have the highest impact on model performance, run the following command:
```Bash
python estimate_impact.py \
    --base_model <model to be estimated, e.g. NousResearch/Llama-2-7b-hf> \
    --data_path <processed calibration dataset, e.g. data/processed_data//c4_processed_50k> \
    --output_file <produced cherry indices file, e.g. data/cherry_indices/llama2-7b-impact.pt>
```

## Training

For all LLM scales (7B, 13B), and both base models and chat models (LLaMA2, Vicuna-v1.5), we train the models on a single node with 8 x A100 80GiB GPUs. We use a total batch size of 128, a learning rate of 2e-5, a weight decay of 0.0, a cosine scheduler with 5% warm-up steps. The rest of hyperparameter used in training across different models are as follows:

| Model & bit | num_epochs | micro_batch_size | gradient_accumulation_steps | min_warmup_ratio |
| :--- | :---: | :---: | :---: | :---: |
| LLaMA2-7b 3bit | 1 | 8 | 2 | 0.25 |
| LLaMA2-7b 4bit | 1 | 8 | 2 | 0.1 |
| LLaMA2-13b 3bit | 1 | 4 | 4 | 0.25 |
| LLaMA2-13b 4bit | 1 | 4 | 4 | 0.1 |
| Vicuna-v1.5-7b 3bit | 2 | 8 | 2 | 0.25 |
| Vicuna-v1.5-13b 3bit | 2 | 4 | 4 | 0.25 |

Here is a sample training command:
```Bash
bash run_train.sh
```

You may modify `micro_batch_size` and `gradient_accumulation_steps` to better accommodate your machine.


## Evaluation

We provide a script for evaluating the perplexity results on two widely-used corpora: C4 and WikiText-2:
```Bash
python evaluate_perplexity.py \
    <tokenizer name or path, e.g. NousResearch/Llama-2-7b-hf> \
    <directory where the model checkpoints are written>
```
