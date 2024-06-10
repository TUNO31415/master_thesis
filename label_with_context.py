import os
import pandas as pd
import torch
from transformers import pipeline
from huggingface_hub import login
from transformers.pipelines.pt_utils import KeyDataset
from gpt_utils import process_growing_window, llm_input_generator_with_context
import gc
from utils import read_token
import math
paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"
hg_token = read_token("/tudelft.net/staff-umbrella/tunoMSc2023/codes/token.txt")

def main():
    login(hg_token)

    model_ckpt = 'meta-llama/Llama-2-7b-chat-hf' 
    generator = pipeline(
        "text-generation", 
        model=model_ckpt, 
        device_map='auto', 
        tokenizer=model_ckpt, 
        max_new_tokens=500, 
        do_sample=False,
    )

    transcription_folder_path = paco_path + "ConversationAudio/transcription/"
    output_path = paco_path + "RealTimeSIS_with_context_v2/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_list = os.listdir(transcription_folder_path)
    new_file_list = []

    for f in file_list:
        if not f.endswith("csv"):
            continue
        
        f_tmp = f.split(".")[0]
        speaker_one = "_".join(f_tmp.split("_")[2:4])
        speaker_two = "_".join(f_tmp.split("_")[4:6])
        
        output_path_00 = output_path + f"rt_SIS_{speaker_one}_{f}"
        output_path_01 = output_path + f"rt_SIS_{speaker_two}_{f}"

        if os.path.exists(output_path_00) and os.path.exists(output_path_01):
            continue

        new_file_list.append(f)
    
    print(len(new_file_list))

    for transcription_csv in new_file_list:
        if not transcription_csv.endswith("csv"):
            continue
        input_df, speaker00_name, speaker01_name = process_growing_window(transcription_folder_path + transcription_csv)
        batch_id = "_".join(transcription_csv.split("_")[0:2])
        output_path_00 = output_path + f"rt_SIS_{speaker00_name}_{transcription_csv}"
        output_path_01 = output_path + f"rt_SIS_{speaker01_name}_{transcription_csv}"
        
        # CHANGE THE CODE HERE TO USE DIFFERENT PROMPTS
        input00, input01 = llm_input_generator_with_context(input_df, speaker00_name, speaker01_name, batch_id)
        
        output00 = []
        for out in generator(KeyDataset(input00, "prompt")):
            output00.append(out[0]['generated_text'])
            gc.collect()
            torch.cuda.empty_cache()

        outdf00 = pd.DataFrame(output00)
        outdf00.to_csv(output_path_00)
        gc.collect()
        torch.cuda.empty_cache()

        
        output01 = []
        for out in generator(KeyDataset(input01, "prompt")):
            output01.append(out[0]['generated_text'])
            gc.collect()
            torch.cuda.empty_cache()

        outdf01 = pd.DataFrame(output01)
        outdf01.to_csv(output_path_01)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()