import pandas as pd
import subprocess
import os
import re
import json


paco_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/"
list_unique_dyads_file_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/LIST_unique_dyads_and_clean_SELFONLY.csv"
conv_data_combined_final_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/curated_conv_data_combined_ffinal.csv"

# paco_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/"
# list_unique_dyads_file_path = "/Users/taichi/Desktop/master_thesis/LIST_unique_dyads_and_clean_SELFONLY_2.csv"
# conv_data_combined_final_path = "/Users/taichi/Desktop/master_thesis/curated_conv_data_combined_ffinal_2.csv"

def get_audio_from_video(input_path, output_path):
    command = f"ffmpeg -loglevel quiet -i {input_path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
    subprocess.call(command, shell=True)

def combine_audio(audio_1_path, audio_2_path, output_path):
    command = f"ffmpeg -loglevel quiet -i {audio_1_path} -i {audio_2_path} -filter_complex amix=inputs=2:duration=longest {output_path}"
    subprocess.call(command, shell=True)

def audio_file_process():
    df = pd.read_csv(list_unique_dyads_file_path)
    df = df.dropna()
    df = df.drop_duplicates()
    print(len(df))

    for index, row in df.iterrows():
        batch_num = row["conv_rec_self_local"].split("-")[3]
        vid1_name = row["conv_rec_self_local"]
        vid2_name = row["conv_rec_other_local"]
        speaker0_name = row["selfPID"]
        speaker1_name = row["otherPID"]
        video_1_path = f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/Conversations/{batch_num}/{vid1_name}.mp4"
        video_2_path = f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/Conversations/{batch_num}/{vid2_name}.mp4"
        audio_1_path = f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/{batch_num}/{vid1_name}.wav"
        audio_2_path = f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/{batch_num}/{vid2_name}.wav"
        audio_path_combined = f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/Combined/{speaker0_name}_{speaker1_name}.wav"
        
        if not os.path.exists(f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/{batch_num}/"):
            os.makedirs(f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/{batch_num}/")

        if not os.path.exists(audio_1_path):
            get_audio_from_video(video_1_path, audio_1_path)
        
        if not os.path.exists(audio_2_path):
            get_audio_from_video(video_2_path, audio_2_path)
        
        if not os.path.exists(f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/Combined/"):
            os.makedirs(f"/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/Combined/")

        if not os.path.exists(audio_path_combined):
            combine_audio(audio_1_path, audio_2_path, audio_path_combined)

        print(f"{index} / {len(df)}")

    print("DONE")

def creat_df_entry(batch, self, other, row):
    if row.shape[0] != 1:
        if row.shape[0] > 1:
            raise Exception(f'There are multiple rows. \n {row}')
        else:
            raise Exception(f'There is no matching row.')    
    
    conv_SP_SIS_C3 = int(row["conv_SP_SIS_C3"])
    conv_SP_SIS_C4r = 6 - int(row["conv_SP_SIS_C4r"])
    conv_SP_SIS_FD5 = int(row["conv_SP_SIS_FD5"])
    conv_SP_SIS_FD6r = 6 - int(row["conv_SP_SIS_FD6r"])
    conv_SP_SIS_IC7 = int(row["conv_SP_SIS_IC7"])
    conv_SP_SIS_IC8r = 6 - int(row["conv_SP_SIS_IC8r"])
    conv_SP_SIS_MD1 = int(row["conv_SP_SIS_MD1"])
    conv_SP_SIS_MD2r = 6 - int(row["conv_SP_SIS_MD2r"])
    conv_SP_SIS_P10r = 6 - int(row["conv_SP_SIS_P10r"])
    conv_SP_SIS_P9 = int(row["conv_SP_SIS_P9"])

    md = (conv_SP_SIS_MD1 + conv_SP_SIS_MD2r) / 2
    ci = (conv_SP_SIS_C3 + conv_SP_SIS_C4r) / 2
    fi = (conv_SP_SIS_FD5 + conv_SP_SIS_FD6r) / 2
    ic = (conv_SP_SIS_IC7 + conv_SP_SIS_IC8r) / 2
    p = (conv_SP_SIS_P9 + conv_SP_SIS_P10r) / 2
    

    return {'BatchNum' : batch, 'selfPID' : self, 'otherPID' : other, 'MD' : md, 'CI' : ci, 'FI' : fi,'IC' : ic, 'P' : p}

def extract_one_digit_numbers(input_string):
    # Use regular expression to find all one-digit numbers in the input string
    one_digit_numbers = re.findall(r'\b\d\b', input_string)
    
    # Convert the matched strings to integers and return as a list
    one_digit_numbers = [int(num) for num in one_digit_numbers]
    
    return one_digit_numbers

def retrospective_sis_process():
    list_unique_dyads_df = pd.read_csv(list_unique_dyads_file_path)
    list_unique_dyads_df = list_unique_dyads_df.dropna()
    list_unique_dyads_df = list_unique_dyads_df.drop_duplicates()
    conv_data_df = pd.read_csv(conv_data_combined_final_path)
    output_list = []

    for index, row in list_unique_dyads_df.iterrows():
        speaker0_name = row["selfPID"]
        speaker1_name = row["otherPID"]
        batch_num = row["conv_rec_self_local"].split("-")[3]

        transcription_path = paco_path + f"ConversationAudio/transcription/{batch_num}_{speaker0_name}_{speaker1_name}.csv"

        if not os.path.exists(transcription_path):
            continue
        
        matching_row_0 = conv_data_df[(conv_data_df['selfPID'] == speaker0_name) & (conv_data_df['batchID'] == batch_num) &  (conv_data_df['otherPID'] == speaker1_name)]
        matching_row_1 = conv_data_df[(conv_data_df['selfPID'] == speaker1_name) & (conv_data_df['batchID'] == batch_num) &  (conv_data_df['otherPID'] == speaker0_name)]

        retro_sis_0 = creat_df_entry(batch_num, speaker0_name, speaker1_name, matching_row_0)
        retro_sis_1 = creat_df_entry(batch_num, speaker1_name, speaker0_name, matching_row_1)

        output_list.append(retro_sis_0)
        output_list.append(retro_sis_1)
        print(f"DONE --- {index} --- {batch_num} {speaker0_name}+{speaker1_name}")
    
    # output_csv_path = paco_path + "retrospective_sis.csv"
    output_csv_path = "/Users/taichi/Desktop/retrospective_sis.csv"
    df = pd.DataFrame(output_list)
    df.to_csv(output_csv_path)

    print("DONE")

def decode_sis(sis_raw_input):
    conv_SP_SIS_C3 = sis_raw_input[0]
    conv_SP_SIS_C4r = 6 - sis_raw_input[1]
    conv_SP_SIS_FD5 = sis_raw_input[2]
    conv_SP_SIS_FD6r = 6 - sis_raw_input[3]
    conv_SP_SIS_IC7 = sis_raw_input[4]
    conv_SP_SIS_IC8r = 6 - sis_raw_input[5]
    conv_SP_SIS_MD1 = sis_raw_input[6]
    conv_SP_SIS_MD2r = 6 - sis_raw_input[7]
    conv_SP_SIS_P10r = 6 - sis_raw_input[8]
    conv_SP_SIS_P9 = sis_raw_input[9]

    md = (conv_SP_SIS_MD1 + conv_SP_SIS_MD2r) / 2
    ci = (conv_SP_SIS_C3 + conv_SP_SIS_C4r) / 2
    fi = (conv_SP_SIS_FD5 + conv_SP_SIS_FD6r) / 2
    ic = (conv_SP_SIS_IC7 + conv_SP_SIS_IC8r) / 2
    p = (conv_SP_SIS_P9 + conv_SP_SIS_P10r) / 2

    return [md, ci, fi, ic, p]               

def process_real_time_sis():
    # real_time_sis_path = paco_path + "RealTimeSIS/"
    # output_folder_path = paco_path + "RealTimeSIS/score_only/"

    real_time_sis_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v3/"
    output_folder_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v3_score_only/"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file in os.listdir(real_time_sis_path):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(real_time_sis_path + file)
        rows = []
        ouput_path = output_folder_path + f"score_only_{file}"
        invalid_flag = False

        if os.path.exists(ouput_path):
            continue

        for index, row in df.iterrows():
            text = row['0']
            part = text.split("[/INST]", 1)[1]
            numbers_list = extract_one_digit_numbers(part)
            scores =  [int(num) for num in numbers_list]
            scores = scores[-5:]

            if all(scores) > 0 and all(scores) < 6 and len(scores) == 5:
                row = {'index' : index, 'MD' : scores[0], 'CI' : scores[1], 'FI' : scores[2],'IC' : scores[3], 'P' : scores[4]}
                rows.append(row)
            else:
                invalid_flag = True
                break

        if not invalid_flag:
            df = pd.DataFrame(rows)
            df.to_csv(ouput_path)
            print(f"DONE --- score_only_{file} SAVED --- ")
        else:
            print(f"FAILED ---- score_only_{file} INVALID INPUT ---- ")

    print(f"DONE")


def get_real_time_sis_v2(sis_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(sis_folder):
        if not file.endswith("csv"):
            continue
        
        df = pd.read_csv(sis_folder + file)
        rows = []
        ouput_path = output_folder + f"score_only_{file}"
        invalid_flag = False

        for index, row in df.iterrows():
            text = row['0']
            part = text.split("[/INST]", 1)[1]
            scores = []
            try:
            # First attempt: using the json module
                data = json.loads(part)
                scores = list(data.values())
                scores = list(map(int, scores))
            except json.JSONDecodeError:
                # Second attempt: using regular expressions
                part = re.sub(r'Q\d+', '', part)
                numbers = re.findall(r'\b\d\b', part)
                scores = list(map(int, numbers))

            if len(scores) < 10:
                invalid_flag = True
                print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th ROW less than 10, only {len(scores)} elements ---- ")
                break

            decoded_scores = decode_sis(scores)
            if all(decoded_scores) > 0 and all(decoded_scores) < 6:
                row_entry = {'index' : index, 'MD' : decoded_scores[0], 'CI' : decoded_scores[1], 'FI' : decoded_scores[2],'IC' : decoded_scores[3], 'P' : decoded_scores[4]}
                rows.append(row_entry)
            else:
                invalid_flag = True
                print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th ROW decoded out of range---- ")
                break

        if not invalid_flag:
            df = pd.DataFrame(rows)
            df.to_csv(ouput_path) 
            print(f"DONE --- score_only_{file} SAVED --- ")
                

if __name__ == "__main__":
    # audio_file_process()
    # retrospective_sis_process()
    get_real_time_sis_v2("/Users/taichi/Desktop/master_thesis/RealTimeSIS_newprompt/", "/Users/taichi/Desktop/master_thesis/rtsis_new_prompt_v2/")
    # process_real_time_sis()