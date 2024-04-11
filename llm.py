# import torch
# from transformers import pipeline
# from huggingface_hub import login
import pandas as pd
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ
# def load_api_keys():
#     file_path = "hg_token.txt"
#     api_keys = {}
#     with open(file_path, 'r') as file:
#         for line in file:
#             name, key = line.strip().split('=')
#             api_keys[name] = key
#     return api_keys

# hg_api_token = load_api_keys()['hg_api']

# login(hg_api_token)


# model_ckpt = 'meta-llama/Llama-2-7b-chat-hf' 
# generator = pipeline(
#     "text-generation", 
#     model=model_ckpt, 
#     device_map='cpu', 
#     tokenizer=model_ckpt, 
#     max_new_tokens=512, 
#     do_sample=False,
# )
prompt = (
" ”Situational Interdependence” is defined in terms "
" ”Mutual Dependence” : Degree of how much each person’s outcomes are determined by how each person behaves in that situation,"
" ”Conflict of Interest” : Degree to which the behavior that results in the best outcome for one individual results in the worst outcome for the other,"
" ”Future Interdependence” : Degree to which own and others’ behavior in the present situation can affect own and others behavior and outcomes in future interactions"
" ”Information Certainty” : Degree to which a person knows their partner’s preferred outcomes and how each person’s actions influence each other’s outcomes"
" ”Power” : Degree to which an individual determines their own and others’ outcomes, while others do not influence their own outcome"
)

# prompt_template = (
# f"[INST] <<SYS>>\n" 
# "Given the dialogue history between PersonA and PersonB : "
# f"{dialogue} \n"
# f"Analyse the extent of each elements of situational interdependence in the next utterance of PersonA {next_utterance} on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n"
# f"<</SYS>>\n{prompt}[/INST]\n"
# )

# output = generator(prompt_template)
# print(output[0]['generated_text'][len(prompt_template):])

cols = ["start", "end", "text", "speaker"]
transcription_csv = "/Users/taichi/Downloads/output (3).csv"
df = pd.read_csv(transcription_csv, usecols=cols)
conversation = df.values.tolist()

dialogue = []
current_speaker = "SPEAKER_01"

for segment in conversation:
    if segment[3] == current_speaker:
        next_utterance = segment[2]
        if len(dialogue) == 0:
            prompt_template = (
                f"[INST] <<SYS>>\n" 
                f"Analyse the extent of each elements of situational interdependence in the next utterance of {current_speaker} \"{next_utterance}\" on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n"
                f"<</SYS>>\n{prompt}[/INST]\n"
            )
        else:
            prompt_template = (
                f"[INST] <<SYS>>\n" 
                "Given the dialogue history between SPEAKER_01 and SPEAKER_01 : "
                f"{dialogue} \n"
                f"Analyse the extent of each elements of situational interdependence in the next utterance of {current_speaker} \"{next_utterance}\" on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n"
                f"<</SYS>>\n{prompt}[/INST]\n"
            )

    dialogue.append([{"speaker" : segment[3], "text" : segment[2]}])

print(dialogue)