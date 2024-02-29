import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


# Default torch device should be the GPU
torch.set_default_device("cuda")


# Load model in 4-bit quantized and qlora
# Check out the following resource for more info: https://huggingface.co/blog/4bit-transformers-bitsandbytes
# This article also helped with the pipeline: https://medium.com/thedeephub/optimizing-phi-2-a-deep-dive-into-fine-tuning-small-language-models-9d545ac90a99
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Loading Phi-2 model
# Phi-2 main article: https://huggingface.co/microsoft/phi-2
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='microsoft/phi-2',
                                             device_map='auto',
                                             quantization_config=nf4_config,
                                             trust_remote_code=True)

# Setting up the tokenizer for Phi-2
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2',
                                          trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"