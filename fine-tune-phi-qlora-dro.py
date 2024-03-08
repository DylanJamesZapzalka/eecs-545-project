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


# This script is based off of the following two articles:
# https://kaitchup.substack.com/p/phi-2-a-small-model-easy-to-fine
# https://towardsdatascience.com/optimizing-small-language-models-on-a-free-t4-gpu-008c37700d57
# Very efficient way of training phi-2 with qlora. Only uses like 6 GB of memory.





base_model_id = "microsoft/phi-2"

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_eos_token=True, use_fast=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token



# Helper function to format the dataset
def chatml_format(example):
    # Formatting system response
    if len(example['system']) > 0:
        message = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Formatting user instruction
    message = {"role": "user", "content": example['question']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Formatting the chosen answer
    chosen = example['chosen'] + "\n"

    # Formatting the rejected answer
    rejected = example['rejected'] + "\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Loading the dataset
dataset = load_dataset("Intel/orca_dpo_pairs")['train']

# Saving original columns for removal
original_columns = dataset.column_names

# Applying formatting to the dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns
)















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


# Initialize Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    max_steps=100, # we set up the max_steps to 100 for demo purpose
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    save_strategy="no",
    logging_steps=1,
    output_dir="phi-2-stuff",
    optim="paged_adamw_32bit",
    warmup_steps=5,
    remove_unused_columns=False,
)

model.config.use_cache = False


# Initialize DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Start Fine-tuning with DPO
dpo_trainer.train()