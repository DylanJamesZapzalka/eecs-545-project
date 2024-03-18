'''
Script used for training the Phi-2 model with DRO.
QLora is used to reduce memory consumption.
This script was created with the help of many medium articles:
    https://kaitchup.substack.com/p/phi-2-a-small-model-easy-to-fine
    https://towardsdatascience.com/optimizing-small-language-models-on-a-free-t4-gpu-008c37700d57
    https://medium.com/@prasadmahamulkar/fine-tuning-phi-2-a-step-by-step-guide-e672e7f1d009
'''

from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import DPOTrainer
import json
import argparse
import tqdm


PHI_2_MODEL_ID = 'microsoft/phi-2'


# Generate command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    help='Dataset to use -- can be \"segmentation\" or \"generation\"')
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
tokenizer = AutoTokenizer.from_pretrained(PHI_2_MODEL_ID,
                                          add_eos_token=True,
                                          use_fast=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token


# Json data is in an incorrect form -- need to switch "rows" and "columns"
if dataset == 'segmentation':
    json_file = open('./datasets/code_summary_dataset.json')
elif dataset == 'generation':
    json_file = open('./datasets/code_solution_dataset.json')
else:
    raise ValueError('--dataset flag must be \"segmentation\" or \"generation\"')

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
          PHI_2_MODEL_ID, 
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
        target_modules= ["q_proj","k_proj","v_proj","fc2","fc1"]
)


if not is_online:

    # Define the training arguments
    training_arguments = TrainingArguments(
            output_dir='./phi-2-training-results',
            per_device_train_batch_size=1,
            learning_rate=lr,
            optim='paged_adamw_8bit',
            bf16=True, #change to fp16 if are using an older GPU
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

    for epoch in range(num_epochs):

        # If not the first epoch, generate a new dataset with the latest fine-tuned Phi-2 model
        if epoch != 0:
            
            # Load the latest tokenizer/model
            tokenizer = AutoTokenizer.from_pretrained('./trained_models/' + model_name + '_' + str(epoch-1)).to('cuda')
            model = AutoModelForCausalLM.from_pretrained('./trained_models/' + model_name + '_' + str(epoch-1), quantization_config=bnb_config).to('cuda')

            # Load original dataset
            if dataset == 'segmentation':
                json_file = open('./datasets/code_summary_dataset.json')
            elif dataset == 'generation':
                json_file = open('./datasets/code_solution_dataset.json')
            else:
                raise ValueError('--dataset flag must be \"segmentation\" or \"generation\"')

            # Create new dataset for current epoch using latest model
            json_data = json.load(len(json_data['prompt']))
            data_list = []
            print("Generating training set for epoch " + str(epoch))
            for i in tqdm.tqdm(range(10), position=0, leave=True):
                sample = {}
                sample['prompt'] = json_data['prompt'][i]
                sample['chosen'] = json_data['chosen'][i]
                inputs = tokenizer(sample['prompt'], return_tensors="pt", return_attention_mask=False).to('cuda')
                outputs = model.generate(**inputs, max_length=500, pad_token_id=tokenizer.eos_token_id)
                rejected = tokenizer.batch_decode(outputs)[0]
                sample['rejected'] = rejected
                data_list.append(sample)
            json_file.close()

            # Load the transformed dataset
            phi_dataset = Dataset.from_list(data_list)


        # Define the training arguments
        training_arguments = TrainingArguments(
            output_dir='./phi-2-training-results',
            per_device_train_batch_size=1,
            learning_rate=lr,
            optim='paged_adamw_8bit',
            bf16=True, #change to fp16 if are using an older GPU
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
        dpo_trainer.save_model('./trained_models/' + model_name + '_' + str(epoch))