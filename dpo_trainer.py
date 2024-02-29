
'''
https://huggingface.co/docs/trl/main/en/dpo_trainer
`pip install trl` before running code
'''

from trl import DPOTrainer
import torch
from transformers import TrainingArguments

'''
Returns DPOTrainer Object
- model: model to be tuned
- training_args : args used for training (has to be transformer.TrainingArguments)
- train_dataset: preference dataset to be optimized on
    - Format: Dictionary must be structured as below
    {
        "prompt" : [p1, p2, p3, ...],
        "chosen" : [c1, c2, c3, ...],
        "rejected" : [r1, r2, r3, ...]
    }
    - For a given prompt p_i, c_i is the preferred answer and r_i is the rejected answer
- tokenizer: tokenizer of the language model to be trained
'''
def trainer(model, training_args, train_dataset, tokenizer):
    return dpo_trainer = DPOTrainer(
        model,
        model_ref = None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

model = None #model to be trained
tokenizer = None #model variable's associated tokenizer
train_dataset = {"prompt": ["hello"], "chosen": ['hi'], "rejected":['leave me alone']}
training_args =  TrainingArguments(output_dir="./output")
dpo_trainer = trainer()
dpo_trainer.train()

