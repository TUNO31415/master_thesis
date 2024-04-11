# import torch
# from transformers import pipeline
# from huggingface_hub import login
import pandas as pd
from datasets import Dataset
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

cols = ["start", "end", "text", "speaker"]
# transcription_csv = "/kaggle/input/transcription-sample/output (3).csv"
transcription_csv = "/Users/taichi/Downloads/output (3).csv"
# df = pd.read_csv(transcription_csv, usecols=cols)
# conversation = df.values.tolist()

# dialogue = []
# current_speaker = "SPEAKER_01"
# output_list = []
# cnt = 0
# total_len = len(conversation)

prompt = (
" ”Situational Interdependence” is defined in terms "
" ”Mutual Dependence” : Degree of how much each person’s outcomes are determined by how each person behaves in that situation."
" ”Conflict of Interest” : Degree to which the behavior that results in the best outcome for one individual results in the worst outcome for the other."
" ”Future Interdependence” : Degree to which own and others’ behavior in the present situation can affect own and others behavior and outcomes in future interactions."
" ”Information Certainty” : Degree to which a person knows their partner’s preferred outcomes and how each person’s ac- tions influence each other’s outcomes."
" ”Power” : Degree to which an individual determines their own and others’ outcomes, while others do not influence their own outcome. \n"
" Please provide your answer as in the following format"
)



print("start processing")

def process_growing_window(csv_path):
    cols = ["start", "end", "text", "speaker"]
    df = pd.read_csv(csv_path, usecols=cols)
    conversation = df.values.tolist()
    new_cols = ["speaker", "utterance", "dialogue history"]

    speakers = [seg[3] for seg in conversation]
    utterances = [seg[2] for seg in conversation]

    history = []
    for i in range(len(conversation)):
        if i == 0:
            history.append("")

        elif i == 1:
            history.append(f"{conversation[i-1][3]} : \"{conversation[i-1][2]}\"")
        else:
            history.append(history[-1] + ", " + f"{conversation[i-1][3]} : \"{conversation[i-1][2]}\" ")

    df = pd.DataFrame({"speaker" : speakers, "utterance" : utterances, "dialogue history" : history})
    df.to_csv('text/growing_window.csv', index=False)
    return df

def llm_input_generator(df):
    print(df.keys())
    speaker_00 = df[df["speaker"] == "SPEAKER_00"]
    speaker_01 = df[df["speaker"] == "SPEAKER_01"]
    
    prompt00 = []
    prompt01 = []

    for index, row in speaker_00.iterrows():
        if index == 0:
            prompt_template_head = (
                f'[INST] <<SYS>>\n'
                f'Analyse the extent of each elements of situational interdependence in the next utterance of SPEAKER_00 \"{row["utterance"]}\" on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n'
                f'<</SYS>>\n{prompt}[/INST]\n'
            )
            prompt00.append(prompt_template_head)
        else:
            prompt_template = (
                f'[INST] <<SYS>>\n' 
                'Given the dialogue history between SPEAKER_01 and SPEAKER_01 : '
                f'{row["dialogue history"]} \n'
                f'Analyse the extent of each elements of situational interdependence in the next utterance of SPEAKER_00 \"{row["utterance"]}\" on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n'
                f'<</SYS>>\n{prompt}[/INST]\n'
            )
            prompt00.append(prompt_template)

    for index, row in speaker_01.iterrows():
        if index == 0:
            prompt_template_head = (
                f'[INST] <<SYS>>\n'
                f'Analyse the extent of each elements of situational interdependence in the next utterance of SPEAKER_01 \"{row["utterance"]}\" on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n'
                f'<</SYS>>\n{prompt}[/INST]\n'
            )
            prompt01.append(prompt_template_head)
        else:
            prompt_template = (
                f'[INST] <<SYS>>\n'
                'Given the dialogue history between SPEAKER_01 and SPEAKER_01 : '
                f'{row["dialogue history"]} \n'
                f'Analyse the extent of each elements of situational interdependence in the next utterance of SPEAKER_01 \"{row["utterance"]}\" on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n'
                f'<</SYS>>\n{prompt}[/INST]\n'
            )
            prompt01.append(prompt_template)

    df00 = pd.DataFrame({"prompt" : prompt00})
    df01 = pd.DataFrame({"prompt" : prompt01})
    df00.to_csv('text/prompt_00.csv', index=False)
    df01.to_csv('text/prompt_01.csv', index=False)
    ds00 = Dataset.from_pandas(df00)
    ds01 = Dataset.from_pandas(df01)
    return ds00, ds01

dataset = process_growing_window(transcription_csv)
llm_input_generator(dataset)

# for segment in conversation:
#     if segment[3] == current_speaker:
#         next_utterance = segment[2]
#         if len(dialogue) == 0:
#             prompt_template = (
#                 f"[INST] <<SYS>>\n" 
#                 f"Analyse the extent of each elements of situational interdependence in the next utterance of {current_speaker} \"{next_utterance}\" on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n"
#                 f"<</SYS>>\n{prompt}[/INST]\n"
#             )
#             output = generator(prompt_template)
#             output_list.append([{"speaker" : current_speaker, "dialogue history" : [], "utterance" : next_utterance, "output" : output[0]['generated_text'][len(prompt_template):]}])
#             cnt += 1
#             print(f"DONE {cnt} / {total_len}")
            
#         else:
#             prompt_template = (
#                 f"[INST] <<SYS>>\n" 
#                 "Given the dialogue history between SPEAKER_00 and SPEAKER_01 : "
#                 f"{dialogue} \n"
#                 f"Analyse the extent of each elements of situational interdependence in the next utterance of {current_speaker} \"{next_utterance}\" on a scale from 0 to 9, with 0 being \"Extremely low\" and 9 being \"Extremely high\". \n"
#                 f"<</SYS>>\n{prompt}[/INST]\n"
#             )
#             output = generator(prompt_template)
#             output_list.append([{"speaker" : current_speaker, "dialogue history" : dialogue, "utterance" : next_utterance, "output" : output[0]['generated_text'][len(prompt_template):]}])
#             cnt += 1
#             print(f"DONE {cnt} / {total_len}")
#     else:
#         cnt += 1
#         print(f"DONE {cnt} / {total_len} SKIPPED")

#     dialogue.append([{"speaker" : segment[3], "text" : segment[2]}])