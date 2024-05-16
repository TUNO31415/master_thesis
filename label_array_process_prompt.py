import os
import pandas as pd
import torch
from transformers import pipeline
from huggingface_hub import login
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import gc
import math
os.environ['TRANSFORMERS_CACHE'] = "/tmp/tuno/hg_cache/"
paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"
slrun_id = os.environ.get("SLURM_ARRAY_TASK_ID")

def process_growing_window(csv_path):
    cols = ["start", "end", "text", "speaker"]
    df = pd.read_csv(csv_path, usecols=cols)
    conversation = df.values.tolist()

    speakers = [seg[3] for seg in conversation]
    utterances = [seg[2] for seg in conversation]

    history = []
    for i in range(len(conversation)):
        if i == 0:
            history.append("")
        elif i == 1:
            history.append(f"{conversation[i-1][3]} : \"{conversation[i-1][2]}\" \n")
        else:
            history.append(history[-1] + f"{conversation[i-1][3]} : \"{conversation[i-1][2]}\" \n")

    df = pd.DataFrame({"speaker" : speakers, "utterance" : utterances, "dialogue history" : history})

    speaker00_name = '_'.join(csv_path.split("/")[-1].split("_")[2:4])
    speaker01_name = '_'.join([csv_path.split("/")[-1].split("_")[4], csv_path.split("/")[-1].split("_")[-1].split(".")[0]])
    return df, speaker00_name, speaker01_name

def llm_input_generator(df, speaker00_name, speaker01_name):
    base = """"
        Here, you are asked to rate the interaction you just took part in. We are interested in your personal (subjective) impression of the situation. Thus, we ask you to be as honest as possible and describe the situation by using the following scale: \n 
        Strongly disagree =1, Somewhat disagree = 2, Neither agree nor disagree = 3, Somewhat agree = 4, Strongly Agree = 5 \n

        1. What each of us does in this situation affects the other. \n
        2. Our preferred outcomes in this situation are conflicting. \n
        3. How we behave now will have consequences for future outcomes. \n
        4. We both know what the other wants. \n
        5. Whatever each of us does in this situation, our actions will not affect the other's outcome. \n
        6. We can both obtain our preferred outcomes. \n
        7. Our future interactions are not affected by the outcomes of this situation. \n
        8. I don't think the other knows what I want. \n

        For each item, please think of the same conversation and indicate how the following statements describe the specific situation. \n
        Definitely person X = 1 ; Maybe person X = 2, Neither person X nor myself = 3, Maybe myself = 4, Definitely myself = 5 \n

        9. Who do you feel had more power to determine their own outcomes in this situation? \n
        10. Who has the least amount of influence on the outcomes of this situation? \n

        Use the template to answer in JSON format. \n
        {"Q1" : SCORE, "Q2" : SCORE, "Q3" : SCORE, ... , "Q10" : SCORE}
    """

    speaker_00 = df[df["speaker"] == speaker00_name]
    speaker_01 = df[df["speaker"] == speaker01_name]

    prompt00 = []
    prompt01 = []

    for index, row in speaker_00.iterrows():
        if index == 0:
            header = f"""
                Act as Person {speaker00_name}. You are now having a conversation with Person {speaker01_name}. You are asked to answer the following 10 questions at the moment you said {row["utterance"]} at the beginning of the conversation. \n
                From now on, person X means your conversation partner, Person {speaker01_name}. \n
            """
            current_prompt = header + base
            prompt00.append(current_prompt)
        else:
            header = f"""
                Act as Person {speaker00_name}. You are now having a conversation with Person {speaker01_name}. You are asked to answer the following 10 questions at the moment you said {row["utterance"]} given this conversation hisotry. \n
                {row["dialogue history"]}
                From now on, person X means your conversation partner, Person {speaker01_name}. \n
            """
            current_prompt = header + base
            prompt00.append(current_prompt)

    for index, row in speaker_01.iterrows():
        if index == 0:
            header = f"""
                Act as Person {speaker01_name}. You are now having a conversation with Person {speaker00_name}. You are asked to answer the following 10 questions at the moment you said {row["utterance"]} at the beginning of the conversation. \n
                From now on, person X means your conversation partner, Person {speaker00_name}. \n
            """
            current_prompt = header + base
            prompt01.append(current_prompt)
        else:
            header = f"""
                Act as Person {speaker01_name}. You are now having a conversation with Person {speaker00_name}. You are asked to answer the following 10 questions at the moment you said {row["utterance"]} given this conversation hisotry. \n
                {row["dialogue history"]}
                From now on, person X means your conversation partner, Person {speaker00_name}. \n
            """
            current_prompt = header + base
            prompt01.append(current_prompt)

    df00 = pd.DataFrame({"prompt" : prompt00})
    df01 = pd.DataFrame({"prompt" : prompt01})
    ds00 = Dataset.from_pandas(df00)
    ds01 = Dataset.from_pandas(df01)
    return ds00, ds01

def llm_input_generator_old(df, speaker00_name, speaker01_name):
    print(df.keys())
    speaker_00 = df[df["speaker"] == speaker00_name]
    speaker_01 = df[df["speaker"] == speaker01_name]

    prompt = (
    " ”Situational Interdependence” is defined in terms of"
    " ”Mutual Dependence (MD)” : Degree of how much each person’s outcomes are determined by how each person behaves in that situation."
    " ”Conflict of Interest (CI)” : Degree to which the behavior that results in the best outcome for one individual results in the worst outcome for the other."
    " ”Future Interdependence (FI)” : Degree to which own and others’ behavior in the present situation can affect own and others behavior and outcomes in future interactions."
    " ”Information Certainty (IC)” : Degree to which a person knows their partner’s preferred outcomes and how each person’s ac- tions influence each other’s outcomes."
    " ”Power (P)” : Degree to which an individual determines their own and others’ outcomes, while others do not influence their own outcome. \n"
    " Please provide your answer as in the following example:\n"
    " MD:[NUM],CI:[NUM],FI:[NUM],IC:[NUM],P:[NUM]"
    " Only answer the scores without reasoning/descriptions"
    )

    prompt00 = []
    prompt01 = []

    for index, row in speaker_00.iterrows():
        if index == 0:
            prompt_template_head = (
                f'[INST] <<SYS>>\n'
                f'Analyse the extent of each elements of situational interdependence in the next utterance of SPEAKER:{speaker00_name} \"{row["utterance"]}\" on a scale from 1 to 5, with 1 being \"Extremely low\" and 5 being \"Extremely high\". \n'
                f'<</SYS>>\n{prompt}[/INST]\n'
            )
            prompt00.append(prompt_template_head)
        else:
            prompt_template = (
                f'[INST] <<SYS>>\n' 
                f'Given the dialogue history between SPEAKER:{speaker00_name} and SPEAKER:{speaker01_name} : '
                f'{row["dialogue history"]} \n'
                f'Analyse the extent of each elements of situational interdependence in the next utterance of SPEAKER:{speaker00_name} \"{row["utterance"]}\" on a scale from 1 to 5, with 1 being \"Extremely low\" and 5 being \"Extremely high\". \n'
                f'<</SYS>>\n{prompt}[/INST]\n'
            )
            prompt00.append(prompt_template)

    for index, row in speaker_01.iterrows():
        if index == 0:
            prompt_template_head = (
                f'[INST] <<SYS>>\n'
                f'Analyse the extent of each elements of situational interdependence in the next utterance of SPEAKER:{speaker01_name} \"{row["utterance"]}\" on a scale from 1 to 5, with 1 being \"Extremely low\" and 5 being \"Extremely high\". \n'
                f'<</SYS>>\n{prompt}[/INST]\n'
            )
            prompt01.append(prompt_template_head)
        else:
            prompt_template = (
                f'[INST] <<SYS>>\n'
                f'Given the dialogue history between SPEAKER:{speaker00_name} and SPEAKER:{speaker01_name} : '
                f'{row["dialogue history"]} \n'
                f'Analyse the extent of each elements of situational interdependence in the next utterance of SPEAKER:{speaker01_name} \"{row["utterance"]}\" on a scale from 1 to 5, with 1 being \"Extremely low\" and 5 being \"Extremely high\". \n'
                f'<</SYS>>\n{prompt}[/INST]\n'
            )
            prompt01.append(prompt_template)

    df00 = pd.DataFrame({"prompt" : prompt00})
    df01 = pd.DataFrame({"prompt" : prompt01})
    ds00 = Dataset.from_pandas(df00)
    ds01 = Dataset.from_pandas(df01)
    return ds00, ds01


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

    files = os.listdir(transcription_folder_path)

    num_files = len(files)
    num_sets = 5
    files_per_set = math.ceil(num_files / num_sets)
    file_sets = [files[i * files_per_set:(i + 1) * files_per_set] for i in range(num_sets)]

    for transcription_csv in file_sets[slrun_id]:
        if not transcription_csv.endswith("csv"):
            continue
        input_df, speaker00_name, speaker01_name = process_growing_window(transcription_folder_path + transcription_csv)
        input00, input01 = llm_input_generator(input_df, speaker00_name, speaker01_name)

        output_path_00 = output_path + f"rt_SIS_{speaker00_name}_{transcription_csv}"
        output_path_01 = output_path + f"rt_SIS_{speaker01_name}_{transcription_csv}"
        print(output_path_00)
        
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