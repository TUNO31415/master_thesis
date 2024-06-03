import os
import pandas as pd
import torch
from transformers import pipeline
from huggingface_hub import login
from transformers.pipelines.pt_utils import KeyDataset
from gpt_utils import process_growing_window, split_files_into_chunks, llm_input_generator_summary
import gc
from utils import read_token

# os.environ['TRANSFORMERS_CACHE'] = "/tmp/tuno/hg_cache/"
paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"
slrun_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
hg_token = read_token("/tudelft.net/staff-umbrella/tunoMSc2023/codes/token.txt")

# CHANGE THIS IF THE NUMBER OF ARRAY PROCESS CHANGES IN SBATCH
num_array_process = 3

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
    output_path = paco_path + "RealTimeSIS_summary_label/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_sets = split_files_into_chunks(transcription_folder_path, num_array_process)
    file_index = slrun_id - 1
    for transcription_csv in file_sets[file_index]:
        if not transcription_csv.endswith("csv"):
            continue
        input_df, speaker00_name, speaker01_name = process_growing_window(transcription_folder_path + transcription_csv)
        
        # CHANGE THE CODE HERE TO USE DIFFERENT PROMPTS
        input00, input01 = llm_input_generator_summary(input_df, speaker00_name, speaker01_name)

        output_path_00 = output_path + f"summary_SIS_{speaker00_name}_{transcription_csv}"
        output_path_01 = output_path + f"summary_SIS_{speaker01_name}_{transcription_csv}"
        
        if not os.path.exists(output_path_00):
            output00 = []
            for out in generator(KeyDataset(input00, "prompt")):
                output00.append(out[0]['generated_text'])
                gc.collect()
                torch.cuda.empty_cache()

            outdf00 = pd.DataFrame(output00)
            outdf00.to_csv(output_path_00)
            gc.collect()
            torch.cuda.empty_cache()

        if not os.path.exists(output_path_01):
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