import torch
from transformers import pipeline
from huggingface_hub import login
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ
def load_api_keys():
    file_path = "hg_token.txt"
    api_keys = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, key = line.strip().split('=')
            api_keys[name] = key
    return api_keys

hg_api_token = load_api_keys()['hg_api']

login(hg_api_token)


model_ckpt = 'meta-llama/Llama-2-7b-chat-hf' 
generator = pipeline(
    "text-generation", 
    model=model_ckpt, 
    device_map='cpu', 
    tokenizer=model_ckpt, 
    max_new_tokens=512, 
    do_sample=False,
)
prompt = (
"Consider the following list of concepts, called ”Situational Interdependence” : "
"Mutual Dependence : (definition), 
"Conflict of Interest : (definition), .."

)

prompt_template = (
f"[INST] <<SYS>>\n"
"Given the dialogue history between PersonA and PersonB : "
"[PersonA: ..., PersonB: ..., ...],
"Analyse the extent of ”Situational Interdependence” in the next utterance of PersonA ”...” on a scale from 0 to 9, with 0 being \"Not relevant at all\" and 9 being \"Extremely relevant\". \n"
f"<</SYS>>\n{prompt}[/INST]\n"
)

output = generator(prompt_template)
print(output[0]['generated_text'][len(prompt_template):])