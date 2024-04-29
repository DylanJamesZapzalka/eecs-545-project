from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel,PhiForCausalLM
import torch
import json
from tqdm import tqdm
import os
import argparse

from IPython.display import display
from IPython.display import Markdown
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def to_markdown(text):
  """
  show text in markdown format
  """
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',
                    type=str,
                    help='dir to trained model')
parser.add_argument('--dataset', type = str)
parser.add_argument('--trials', type = int, default = 3)
parser.add_argument('--trial_num', type = int, default = 0)
args = parser.parse_args()
input_dir = args.input_dir
dataset = args.dataset
trials = args.trials
trial_num = args.trial_num


torch.set_default_device("cuda")
#model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True, use_fast=True)
#model = AutoModel.from_pretrained('/home/bswang/Courses/EECS545/eecs-545-project/phi-2-training-results/checkpoint-7000')
model = PhiForCausalLM.from_pretrained(input_dir, token = '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').to('cuda')

if(os.path.exists(os.path.join(input_dir, f'result.json'))):
  json_file = open(os.path.join(input_dir, f'result.json'))
elif(dataset == "generation"):
  json_file = open('./datasets/code_solution_dataset.json')
elif(dataset == "summarization"):
  json_file = open('./datasets/code_summary_dataset.json')
else:
  raise Exception('please double check your dataset input')
json_data = json.load(json_file)
datalist = []

for i in range(len(json_data['prompt'])):
  datalist.append(json_data['prompt'][i])

start_idx = 0
output_text = [] if 'trained_output' not in json_data else json_data['trained_output']
if(len(output_text) == 0 or len(output_text) % (len(datalist) // trials) == 0):
  ratio = len(output_text) / len(datalist)
  ratio += 1/trials * trial_num
  start_idx, end_idx = int(ratio * len(datalist)), int((ratio + 1/trials) * len(datalist))
  end_idx = min(end_idx, len(datalist))
else:
  raise Exception('Please check difference that causes incorrect size')

print(f'start = {start_idx}, end = {end_idx}')
if(start_idx == len(datalist)): exit()
for i in tqdm(range(start_idx, end_idx)):
  try:
    prompt = datalist[i]
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to('cuda')
    
    if(args.dataset == "generation"):
      outputs = model.generate(**inputs, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
    else: #summarization
      outputs = model.generate(**inputs, max_length=len(prompt)+300, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.batch_decode(outputs)[0]
    output_text.append(text)
    
  except: #if error arises (3 samples in dataset)
    output_text.append(None)

  if(i == (start_idx + end_idx) // 2): #save halfway in case the process was way slower than expected
    json_data['trained_output'] = output_text
    with open(os.path.join(input_dir,f'result_{start_idx}.json'), 'w') as f:
      json.dump(json_data, f)

print('answer generated')
json_data['trained_output'] = output_text
with open(os.path.join(input_dir,f'result_{start_idx}.json'), 'w') as f:
    json.dump(json_data, f)
#to_markdown(text)