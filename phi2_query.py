from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  """
  show text in markdown format
  """
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

prompt = "How are you?"
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to('cuda')

outputs = model.generate(**inputs, max_length=500, pad_token_id=tokenizer.eos_token_id)

text = tokenizer.batch_decode(outputs)[0]
to_markdown(text)