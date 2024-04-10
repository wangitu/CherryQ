import os
from typing import Union, Optional

import torch
import transformers
from datasets import load_from_disk

from cherryq.core.cherry_linear import QuantLinear
from cherryq.models.llama.configuration_llama_quant import LlamaConfig
from cherryq.models.llama.modeling_llama_quant import LlamaForCausalLM
from cherryq.training.trainer import Trainer
import utils

logger = utils.get_logger(__name__)


def train(
    # model/data params
    base_model,
    data_path,
    output_dir,
    
    # training hyperparams
    num_epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    save_steps: int,
    learning_rate: float,
    lr_scheduler_type: str,
    weight_decay: float,
    warmup_ratio: float,
    min_warmup_ratio: float,
    fsdp: str = '',
    fsdp_config: Optional[str] = None,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,  # either training checkpoint or final adapter
    
    # wandb params
    wandb_project: str = '',
    wandb_run_name: str = '',
    
    # QAT params
    w_bits: int = 3,
    group_size: int = 64,
    
    # cherry params
    cherry_indices_file: Optional[str] = None, # Naive QAT if `cherry_indices_file` is None, otherwise CherryQ
    cherry_fraction: float = 1/256,
):  
    device_id = int(os.environ.get('LOCAL_RANK', 0))
    train_data = load_from_disk(data_path)
    
    
    logger.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model, padding_side="right", use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True
    )
    logger.info("Complete tokenizer loading...")
    
    
    logger.info("Start to load model...")
    config: LlamaConfig = LlamaConfig.from_pretrained(base_model)
    config.pretraining_tp = 1
    config.w_bits = w_bits
    config.group_size = group_size
    config.cherryq = cherryq = cherry_indices_file is not None
    config.cherry_fraction = cherry_fraction
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model,
        config=config,
        attn_implementation="flash_attention_2",
        device_map={'': device_id},
    )
    logger.info(model.config)
    
    if hasattr(model, "post_init_"):
        model.post_init_() # post init here to avoid memory leakage when `teacher_model` is loaded after `model.post_init_`. I don't know why memory leakage occurs, maybe PyTorch's bug.
    model.config.use_cache = False
    model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.do_sample = True # avoid throwing warnings in `validate`.
    
    
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
        
    
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    fp16 = not bf16
    training_args = transformers.trainer.TrainingArguments(
        output_dir=output_dir,
        seed=42,
        data_seed=42,
        do_train=True,
        num_train_epochs=num_epochs,
        optim="adamw_torch",
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        half_precision_backend="auto",
        fp16=fp16,
        bf16=bf16,
        adam_beta1=0.9,
        adam_beta2=0.95,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        logging_steps=1,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    if cherryq:
        cherry_indices_mapping = torch.load(cherry_indices_file)
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.register_cherry_indices(cherry_indices_mapping[name])
                
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=collator,
        min_warmup_ratio=min_warmup_ratio
    )
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    import fire
    
    fire.Fire(train)
    