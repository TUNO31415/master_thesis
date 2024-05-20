import os
import pandas as pd
import torch
from transformers import pipeline
from huggingface_hub import login
from transformers.pipelines.pt_utils import KeyDataset
import gc
from gpt_utils import process_growing_window, llm_input_generator
os.environ['TRANSFORMERS_CACHE'] = "/tmp/tuno/hg_cache/"
paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"

def main():
    login("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")

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
    output_path = paco_path + "RealTimeSIS_v2/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for transcription_csv in os.listdir(transcription_folder_path):
        if not transcription_csv.endswith("csv"):
            continue

        input_df, speaker00_name, speaker01_name = process_growing_window(transcription_folder_path + transcription_csv)
        input00, input01 = llm_input_generator(input_df, speaker00_name, speaker01_name)

        output_path_00 = output_path + f"rt_SIS_{speaker00_name}_{transcription_csv}"
        output_path_01 = output_path + f"rt_SIS_{speaker01_name}_{transcription_csv}"
        
        if not os.path.exists(output_path_00):
            output00 = []
            for out in generator(KeyDataset(input00, "prompt")):
                gc.collect()
                torch.cuda.empty_cache()
                output00.append(out[0]['generated_text'])

            outdf00 = pd.DataFrame(output00)
            outdf00.to_csv(output_path_00)
            gc.collect()
            torch.cuda.empty_cache()

        if not os.path.exists(output_path_01):
            output01 = []
            for out in generator(KeyDataset(input01, "prompt")):
                gc.collect()
                torch.cuda.empty_cache()
                output01.append(out[0]['generated_text'])

            outdf01 = pd.DataFrame(output01)
            outdf01.to_csv(output_path_01)
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()