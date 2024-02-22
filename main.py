import llm
# import google.generativeai as genai
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2Model
import torch
import os


# https://huggingface.co/docs/transformers/main/en/installation
save_path = "./models/t5-small"
save_path_GPT2 = "./models/gpt2"

def save_T5():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print("t5 is saved successfully")
    return save_path

def load_T5():
    tokenizer = T5Tokenizer.from_pretrained(save_path)
    model = T5ForConditionalGeneration.from_pretrained(save_path)
    return model, tokenizer


def save_GPT2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer.save_pretrained(save_path_GPT2)
    model.save_pretrained(save_path_GPT2)
    return save_path_GPT2

def load_GPT2():
    tokenizer = GPT2Tokenizer.from_pretrained(save_path_GPT2)
    model = GPT2Model.from_pretrained(save_path_GPT2)
    return model, tokenizer
    
def query(model, tokenizer, query):
    input_id = tokenizer([query], return_tensors="pt").input_ids
    output = model.generate(input_id)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

def query_g(model, tokenizer, query):
    encoded_input = tokenizer(query, return_tensors='pt')
    output = model(**encoded_input)
    return output

# save_path = save_T5()
# model, tokenizer = load_T5()
# output = query(model, tokenizer, "answer this question: explain Newton")
# print(output)
save_GPT2()
model, tokenizer = load_GPT2()
output = query_g(model, tokenizer, "explain newton")
decoded_output = tokenizer.decode(output[0])
print(decoded_output)


