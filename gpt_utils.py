import pandas as pd
from datasets import Dataset
import os
import prompt_creation

def process_growing_window(csv_path):
    cols = ["start", "end", "text", "speaker"]
    df = pd.read_csv(csv_path, usecols=cols)
    conversation = df.values.tolist()

    speakers = [seg[3] for seg in conversation]
    utterances = [seg[2] for seg in conversation]

    history = []
    for i in range(len(conversation)):
        if i == 0:
            history.append(f"{conversation[i][3]} : \"{conversation[i][2]}\" \n")
        elif i == 1:
            history.append(f"{conversation[i][3]} : \"{conversation[i][2]}\" \n")
        else:
            history.append(history[-1] + f"{conversation[i][3]} : \"{conversation[i][2]}\" \n")

    df = pd.DataFrame({"speaker" : speakers, "utterance" : utterances, "dialogue history" : history})

    speaker00_name = '_'.join(csv_path.split("/")[-1].split("_")[2:4])
    speaker01_name = '_'.join([csv_path.split("/")[-1].split("_")[4], csv_path.split("/")[-1].split("_")[-1].split(".")[0]])
    return df, speaker00_name, speaker01_name

def llm_input_generator_without_number(df, speaker00_name, speaker01_name):
    return prompt_creation.llm_input_generator_without_number(df, speaker00_name, speaker01_name)

def llm_input_generator_per_question(df, speaker00_name, speaker01_name):
    return prompt_creation.llm_input_generator_per_questions(df, speaker00_name, speaker01_name)

def llm_input_generator(df, speaker00_name, speaker01_name):
    return prompt_creation.llm_input_generator(df, speaker00_name, speaker01_name)

def llm_input_generator_summary(df, speaker00_name, speaker01_name):
    return prompt_creation.llm_input_generator_summary(df, speaker00_name, speaker01_name)

def split_files_into_chunks(directory_path, num_chunks):
    # Get list of files in the directory
    files = os.listdir(directory_path)
    
    # Calculate number of files per chunk and any remainder
    num_files = len(files)
    files_per_chunk = num_files // num_chunks
    remainder = num_files % num_chunks
    
    # Initialize a list of chunks to hold the divided files
    chunks = []
    start = 0
    
    for i in range(num_chunks):
        chunk_size = files_per_chunk + (1 if i < remainder else 0)
        chunks.append(files[start:start+chunk_size])
        start += chunk_size
    
    return chunks

def exist_output(transcription_path, output_folder):
    speaker00_name = '_'.join(transcription_path.split("/")[-1].split("_")[2:4])
    speaker01_name = '_'.join([transcription_path.split("/")[-1].split("_")[4], transcription_path.split("/")[-1].split("_")[-1].split(".")[0]])

    output_path_00 = output_folder + f"rt_SIS_{speaker00_name}_{transcription_path}"
    output_path_01 = output_folder + f"rt_SIS_{speaker01_name}_{transcription_path}"

    if os.exists(output_path_00) and os.exists(output_path_01):
        return True
    else:
        return False