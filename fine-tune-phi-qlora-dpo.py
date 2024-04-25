'''
Script used for training the Phi-2 model with DPO.
QLora is used to reduce memory consumption.
This script was created with the help of many medium articles:
    https://kaitchup.substack.com/p/phi-2-a-small-model-easy-to-fine
    https://towardsdatascience.com/optimizing-small-language-models-on-a-free-t4-gpu-008c37700d57
    https://medium.com/@prasadmahamulkar/fine-tuning-phi-2-a-step-by-step-guide-e672e7f1d009
'''

import argparse
import gc
import json
import os

import torch
import tqdm
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import DPOTrainer
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

MODEL_ID = 'google/gemma-2b-it'

# Generate command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    help='Dataset to use -- can be \"summarization\" or \"generation\"')
parser.add_argument('--model_name',
                    type=str,
                    help='Name of the model.')
parser.add_argument('--is_online',
                    action='store_true',
                    help='Indicates whether or not to use online distillation or not.')
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    help='Learning rate to use.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=3,
                    help='Number of epochs to train for.')

# Parse command line arguments
args = parser.parse_args()
dataset = args.dataset
model_name = args.model_name
is_online = args.is_online
lr = args.lr
num_epochs = args.num_epochs

# Load the Phi-2 tokenizer and define how the tokens should be padded
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                          add_eos_token=True,
                                          use_fast=True)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# Json data is in an incorrect form -- need to switch "rows" and "columns"
if dataset == 'summarization':
    json_file = open('./datasets/code_summary_dataset.json')
elif dataset == 'generation':
    json_file = open('./datasets/code_solution_dataset.json')
else:
    raise ValueError('--dataset flag must be \"summarization\" or \"generation\"')

json_data = json.load(json_file)
data_list = []
for i in range(len(json_data['prompt'])):
    sample = {}
    sample['prompt'] = json_data['prompt'][i]
    sample['chosen'] = json_data['chosen'][i]
    sample['rejected'] = json_data['rejected'][i]
    data_list.append(sample)
json_file.close()

# Load the transformed dataset
phi_dataset = Dataset.from_list(data_list)

# Load the model with 4bit quauntization
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype='bfloat16',
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    load_in_4bit=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# Define the QLORA settings
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "fc2", "fc1"]
)

if not is_online:

    # Define the training arguments
    training_arguments = TrainingArguments(
        output_dir=f'./{model_name}-training-results',
        per_device_train_batch_size=1,
        learning_rate=lr,
        optim='paged_adamw_8bit',
        bf16=True,  # change to fp16 if are using an older GPU
        num_train_epochs=num_epochs,
        report_to=None)

    # Create the DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        train_dataset=phi_dataset,
        peft_config=peft_config,
        max_prompt_length=512,
        max_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # Start the process of fine-tuning Phi-2 with DPO
    dpo_trainer.train()

    # Save the fine-tuned model
    dpo_trainer.save_model('./trained_models/' + model_name)


elif is_online:
    start_epoch = 0
    for i in range(3, -1, -1):
        if os.path.isdir(f'./trained_models/{model_name}_{dataset}_{i}'):
            start_epoch = i + 1
            break

    for epoch in range(num_epochs)[start_epoch:]:

        # If not the first epoch, generate a new dataset with the latest fine-tuned Phi-2 model
        if epoch != 0:

            # Load the latest tokenizer/model
            model_path = f'./trained_models/{model_name}_{dataset}_{epoch - 1}'
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                         quantization_config=bnb_config)

            # Load original dataset
            if dataset == 'summarization':
                json_file = open('./datasets/code_summary_dataset.json')
            elif dataset == 'generation':
                json_file = open('./datasets/code_solution_dataset.json')
            else:
                raise ValueError('--dataset flag must be \"summarization\" or \"generation\"')

            # Create new dataset for current epoch using latest model
            json_data = json.load(json_file)
            data_list = []
            print("Generating training set for epoch " + str(epoch))
            for i in tqdm.tqdm(range(len(json_data['prompt'])), position=0, leave=True):
                sample = {}
                sample['prompt'] = json_data['prompt'][i]
                sample['chosen'] = json_data['chosen'][i]
                inputs = tokenizer(sample['prompt'], return_tensors="pt", return_attention_mask=False).to('cuda')
                outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
                rejected = tokenizer.batch_decode(outputs)[0]
                sample['rejected'] = rejected
                data_list.append(sample)
            json_file.close()

            # Load the transformed dataset
            phi_dataset = Dataset.from_list(data_list)

        # Define the training arguments
        training_arguments = TrainingArguments(
            output_dir=f'./{model_name}-training-results',
            per_device_train_batch_size=1,
            learning_rate=lr,
            optim='paged_adamw_8bit',
            bf16=True,  # change to fp16 if are using an older GPU
            num_train_epochs=1,
            report_to=None
        )

        # Create the DPO Trainer
        dpo_trainer = DPOTrainer(
            model=model,
            train_dataset=phi_dataset,
            peft_config=peft_config,
            max_prompt_length=512,
            max_length=1024,
            tokenizer=tokenizer,
            args=training_arguments,
        )

        # Start the process of fine-tuning Phi-2 with DPO
        dpo_trainer.train()

        # Save the fine-tuned model
        dpo_trainer.save_model('./trained_models/' + model_name + '_' + dataset + '_' + str(epoch))

        # Free memory
        torch._C._cuda_clearCublasWorkspaces()
        torch._dynamo.reset()
        del tokenizer
        del model
        del dpo_trainer
        gc.collect()
        torch.cuda.empty_cache()
