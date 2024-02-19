import llm
# import google.generativeai as genai
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


# https://huggingface.co/docs/transformers/main/en/installation

save_path = "./models/t5-small"

def save_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

def load_model():
    tokenizer1 = T5Tokenizer.from_pretrained(save_path)
    model1 = T5ForConditionalGeneration.from_pretrained(save_path)
    input_id = tokenizer1(["translate English to German: The house is wonderful."], return_tensors="pt").input_ids
    output = model1.generate(input_id)
    output = tokenizer1.decode(output[0], skip_special_tokens=True)
    return output

print(load_model())
# hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir=save_path)