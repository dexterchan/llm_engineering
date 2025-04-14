# %%
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
# %%
import os
hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
# %%
LLAMA = "mlx-community/Llama-3.1-8B-Instruct-4bit"

# %%
model_name = LLAMA
from mlx_lm import load, generate, stream_generate
model, tokenizer = load(model_name)

prompt = "Hello, how can I assist you today?"
response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
# %%
response
# %%
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]

# %%
# Custom chat template
def apply_mlx_chat_template(chat, add_generation_prompt=False):
    prompt = "<|begin_of_text|>"
    for message in chat:
        role = message["role"]
        content = message["content"]
        if role == "user":
            prompt += f"<|start_header_id|>user<|end_header_id|> {content}<|eot_id|>"
        elif role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|> {content}<|eot_id|>"
        elif role == "system":
            prompt += f"<|start_header_id|>system<|end_header_id|>{content}<|eot_id|>"
        else:
            raise ValueError("Role must be 'user' or 'assistant'")
    if add_generation_prompt:
        prompt += "<|start_header_id|>assistant<|end_header_id|>"
    return prompt
# %%
# Format and generate
formatted_prompt = tokenizer.apply_chat_template(messages,add_generation_prompt=True)
# response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=3000)
# print("Response:\n", response.text)

for response in stream_generate(model, tokenizer, formatted_prompt, max_tokens=512):
    print(response.text, end="", flush=True)
# %%