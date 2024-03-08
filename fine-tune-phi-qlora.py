# Importing packages
import os
import gc
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import DPOTrainer
import bitsandbytes as bnb
from trl import SFTTrainer


# This script is based off of https://kaitchup.substack.com/p/phi-2-a-small-model-easy-to-fine
# Very efficient way of training phi-2 with qlora. Only uses like 6 GB of memory.


base_model_id = "microsoft/phi-2"

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_eos_token=True, use_fast=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          base_model_id, trust_remote_code=True, quantization_config=bnb_config, device_map={"": 0}, torch_dtype="auto", attn_implementation="flash_attention_2"
)

model = prepare_model_for_kbit_training(model)
dataset = load_dataset("timdettmers/openassistant-guanaco")

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj","k_proj","v_proj","fc2","fc1"]
)

training_arguments = TrainingArguments(
        output_dir="./phi2-results2",
        evaluation_strategy="steps",
        do_eval=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=12,
        per_device_eval_batch_size=1,
        log_level="debug",
        save_steps=100,
        logging_steps=25, 
        learning_rate=1e-4,
        eval_steps=50,
        optim='paged_adamw_8bit',
        bf16=True, #change to fp16 if are using an older GPU
        num_train_epochs=3,
        warmup_steps=100,
        lr_scheduler_type="linear",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=True
)

trainer.train()