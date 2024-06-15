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
    # conv_SP_SIS_C3 = sis_raw_input[0]
    # conv_SP_SIS_C4r = 6 - sis_raw_input[1]
    # conv_SP_SIS_FD5 = sis_raw_input[2]
    # conv_SP_SIS_FD6r = 6 - sis_raw_input[3]
    # conv_SP_SIS_IC7 = sis_raw_input[4]
    # conv_SP_SIS_IC8r = 6 - sis_raw_input[5]
    # conv_SP_SIS_MD1 = sis_raw_input[6]
    # conv_SP_SIS_MD2r = 6 - sis_raw_input[7]
    # conv_SP_SIS_P10r = 6 - sis_raw_input[8]
    # conv_SP_SIS_P9 = sis_raw_input[9]

    conv_SP_SIS_MD1 = sis_raw_input[0]
    conv_SP_SIS_C3 = sis_raw_input[1]
    conv_SP_SIS_FD5 = sis_raw_input[2]
    conv_SP_SIS_IC7 = sis_raw_input[3]
    conv_SP_SIS_MD2r = 6 - sis_raw_input[4]
    conv_SP_SIS_C4r = 6 - sis_raw_input[5]
    conv_SP_SIS_FD6r = 6 - sis_raw_input[6]
    conv_SP_SIS_IC8r = 6 - sis_raw_input[7]
    conv_SP_SIS_P9 = sis_raw_input[8]
    conv_SP_SIS_P10r = 6 - sis_raw_input[9]

    md = (conv_SP_SIS_MD1 + conv_SP_SIS_MD2r) / 2
    ci = (conv_SP_SIS_C3 + conv_SP_SIS_C4r) / 2
    fi = (conv_SP_SIS_FD5 + conv_SP_SIS_FD6r) / 2
    ic = (conv_SP_SIS_IC7 + conv_SP_SIS_IC8r) / 2
    p = (conv_SP_SIS_P9 + conv_SP_SIS_P10r) / 2

    return [md, ci, fi, ic, p]

def process_real_time_sis():
    # real_time_sis_path = paco_path + "RealTimeSIS/"
    # output_folder_path = paco_path + "RealTimeSIS/score_only/"

    real_time_sis_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_without_number/"
    output_folder_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_without_number/score_only/"

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

def get_summary_sis(sis_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(sis_folder):
        if not file.endswith("csv"):
            continue
        
        df = pd.read_csv(sis_folder + file)
        rows = []
        ouput_path = output_folder + f"score_only_{file}"
        invalid_flag = False

        mapping = {
            "strongly disagree": 1,
            "definitely disagree" : 1,
            "somewhat disagree": 2,
            "neither agree": 3,
            "somewhat agree": 4,
            "strongly agree": 5,
            "definitely agree": 5,
            "definitely person": 1,
            "maybe person": 2,
            "neither person": 3,
            "maybe myself": 4,
            "definitely myself": 5,
            "person x" : 1
        }

        for index, row in df.iterrows():
            text = row['0']
            part = text.split("[/INST]", 1)[1]
            scores = []
            try:
            # First attempt: using the json module
                data = json.loads(part)
                scores = list(data.values())
                try:
                    scores = list(map(int, scores))
                except:
                    try:
                        scores = [re.sub(r'PID_\d+', 'X', a) for a in scores]
                        scores = [mapping[" ".join(a.split(" ")[0:2]).lower()] for a in scores]
                    except:
                        invalid_flag = True
                        print(scores)
                        print([mapping[" ".join(a.split(" ")[0:2]).lower()] for a in scores])
                        print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th Weird input {part}---- ")
                        break
                    
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
                print(scores)
                print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th ROW decoded out of range---- ")
                break

        if not invalid_flag:
            df = pd.DataFrame(rows)
            df.to_csv(ouput_path) 
            # print(f"DONE --- score_only_{file} SAVED --- ")

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

        mapping = {
            "strongly disagree": 1,
            "definitely disagree" : 1,
            "somewhat disagree": 2,
            "neither agree": 3,
            "somewhat agree": 4,
            "strongly agree": 5,
            "definitely agree": 5,
            "definitely person": 1,
            "maybe person": 2,
            "neither person": 3,
            "maybe both": 3,
            "maybe myself": 4,
            "definitely myself": 5,
            "definitely self": 5,
            "person x" : 1
        }

        for index, row in df.iterrows():
            text = row['0']
            part = text.split("[/INST]", 1)[1]
            scores = []
            try:
            # First attempt: using the json module
                data = json.loads(part)
                scores = list(data.values())
                try:
                    scores = list(map(int, scores))
                except:
                    try:
                        scores = [re.sub(r'PID_\d+', 'X', a) for a in scores]
                        scores = [mapping[" ".join(a.split(" ")[0:2]).lower()] for a in scores]
                    except:
                        invalid_flag = True
                        print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th Weird input {part}---- ")
                        break
            except json.JSONDecodeError:
                # Second attempt: using regular expressions
                part = re.sub(r'Q\d+', '', part)
                numbers = re.findall(r'\b\d\b', part)
                scores = list(map(int, numbers))

            # print(scores)

            if len(scores) < 10:
                invalid_flag = True
                print(f"FAILED ---- summary_score_only_{file} INVALID INPUT AT {index}th ROW less than 10, only {len(scores)} elements ---- ")
                break

            decoded_scores = decode_sis(scores)
            if all(decoded_scores) > 0 and all(decoded_scores) < 6:
                row_entry = {'index' : index, 'MD' : decoded_scores[0], 'CI' : decoded_scores[1], 'FI' : decoded_scores[2],'IC' : decoded_scores[3], 'P' : decoded_scores[4]}
                rows.append(row_entry)
            else:
                invalid_flag = True
                print(scores)
                print(f"FAILED ---- summary_score_only_{file} INVALID INPUT AT {index}th ROW decoded out of range---- ")
                break

        if not invalid_flag:
            df = pd.DataFrame(rows)
            df.to_csv(ouput_path) 
            # print(f"DONE --- summary_score_only_{file} SAVED --- ")

def get_real_time_sis_per_question(sis_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mapping = {
        "strongly disagree": 1,
        "definitely disagree" : 1,
        "somewhat disagree": 2,
        "neither agree": 3,
        "somewhat agree": 4,
        "strongly agree": 5,
        "definitely agree": 5,
        "definitely person": 1,
        "maybe person": 2,
        "neither person": 3,
        "maybe myself": 4,
        "definitely myself": 5,
        "person x" : 1
    }


    for file in os.listdir(sis_folder):
        if not file.endswith("csv"):
            continue
        
        df = pd.read_csv(sis_folder + file)
        rows = []
        ouput_path = output_folder + f"score_only_{file}"
        invalid_flag = False

        scores = []
        rows = []

        id = 0

        for index, row in df.iterrows():
            if invalid_flag:
                break

            text = row['0']
            part = text.split("[/INST]", 1)[1]
            label = find_substring_in_list(re.sub(r'PID_\d+', 'X', part).lower(), mapping.keys())

            if label is None:
                invalid_flag = True
                print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th Weird input {part}---- ")
                break
            
            scores.append(mapping[label])
            
            if len(scores) == 10:
                decoded_scores = decode_sis(scores)
                row_entry = {'index' : id, 'MD' : decoded_scores[0], 'CI' : decoded_scores[1], 'FI' : decoded_scores[2],'IC' : decoded_scores[3], 'P' : decoded_scores[4]}
                rows.append(row_entry)
                scores = []
                id += 1

        if not invalid_flag:
            if len(rows) < 1:
                continue
            df = pd.DataFrame(rows)
            df.to_csv(ouput_path) 
            # print(f"DONE --- score_only_{file} SAVED --- ")
            
def find_substring_in_list(input_string, string_list):
    for i in range(len(input_string)):
        for j in range(i + 1, len(input_string) + 1):
            substring = input_string[i:j]
            if substring in string_list:
                return substring
    return None

def get_real_time_sis_without_number(sis_folder, output_folder):
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
            responses = None

            if "'" in part:
                part.replace("'", '"')
            
            json_match = re.search(r'\{.*\}', part, re.DOTALL)
            
            if json_match:
                responses = json.loads(json_match.group(0))

            if responses is None or len(responses) < 10:
                fixed_json = part.replace("'", '"', 1)[1:-1]

                # Split the string into individual question-answer pairs
                parts = fixed_json.split("\\n\\n")

                # Initialize an empty dictionary to hold the results
                responses = {}

                # Populate the dictionary with the split pairs
                for p in parts:
                    key_value = p.split(": ", 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        responses[key.strip()] = value.strip()
                    else:
                        invalid_flag = True
                        print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th Weird input {part}---- ")
                        break

            if responses is None or len(responses) < 10:
                lines = part.strip().split('\n')
                responses = {}

                # Process each line to extract the question and answer
                for line in lines:
                    if line.startswith('Q'):
                        key_value = line.split(': ', 1)
                        if len(key_value) == 2:
                            key, value = key_value
                            responses[key.strip()] = value.strip()
                        else:
                            invalid_flag = True
                            print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th Weird input {part}---- ")
                            break

            mapping = {
                "strongly disagree": 1,
                "definitely disagree" : 1,
                "somewhat disagree": 2,
                "neither agree": 3,
                "somewhat agree": 4,
                "strongly agree": 5,
                "definitely agree": 5,
                "definitely person": 1,
                "maybe person": 2,
                "neither person": 3,
                "maybe myself": 4,
                "definitely myself": 5,
                "person x" : 1
            }

            # Create a list to store the values from Q1 to Q10
            scores = []
            for response in responses.values():
                if type(response) is int:
                    scores.append(response)
                else:
                    try:
                        response = " ".join(response.split(" ")[0:2]).lower()
                        if response in mapping:
                            scores.append(mapping[response])

                    except:
                        invalid_flag = True
                        print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th Weird input ---- ")
                        break

            if len(scores) < 10:
                invalid_flag = True
                print(f"scores : {scores}")
                print(responses)
                print(responses.keys())
                print(responses.values())
                print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th ROW less than 10, only {len(scores)} elements ---- ")
                break

            decoded_scores = decode_sis(scores)
            if all(decoded_scores) > 0 and all(decoded_scores) < 6:
                row_entry = {'index' : index, 'MD' : decoded_scores[0], 'CI' : decoded_scores[1], 'FI' : decoded_scores[2],'IC' : decoded_scores[3], 'P' : decoded_scores[4]}
                rows.append(row_entry)
            else:
                invalid_flag = True
                print(f"output : {scores}")
                print(f"FAILED ---- score_only_{file} INVALID INPUT AT {index}th ROW decoded out of range---- ")
                break

        if not invalid_flag:
            if len(rows) < 1:
                continue
            df = pd.DataFrame(rows)
            df.to_csv(ouput_path) 
            # print(f"DONE --- score_only_{file} SAVED --- ")


def concat_summary_evaluation():
    summary_sis_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_summary_label/score_only/"
    rows = []
    for file in os.listdir(summary_sis_path):
        if not file.endswith(".csv"):
            continue

        # 'summary_SIS_PID_28_Batch_4_PID_28_PID_30.csv'
        
        batch_id = "_".join(file.split("_")[6:8])
        
        self_id = "_".join(file.split("_")[4:6])
        
        other_id = "_".join(file.split("_")[8:10])
        
        
        if self_id == other_id:
            tmp = file.split(".")[0]
            other_id = other_id = "_".join(tmp.split("_")[10:12])
        
        df = pd.read_csv(summary_sis_path + file)

        scores = df.iloc[0].tolist()[2:]
        entry = {
            "BatchNum" : batch_id,
            "selfPID" : self_id,
            "otherPID" : other_id,
            "MD" : scores[0],
            "CI" : scores[1],
            "FI" : scores[2],
            "IC" : scores[3],
            "P" : scores[4]
        }
        rows.append(entry)

    output_path = "/Users/taichi/Desktop/master_thesis/estimated_summary_sis.csv"
    res_df = pd.DataFrame(rows)


    def extract_number(batch_id):
        match = re.search(r'\d+', batch_id)
        return int(match.group()) if match else None

    # Create a new column for the numerical part of batchID
    res_df['batch_number'] = res_df['BatchNum'].apply(extract_number)

    # Step 2: Sort the dataframe based on the numerical part of batchID
    sorted_df = res_df.sort_values(by='batch_number')

    # Drop the temporary 'batch_number' column
    sorted_df = sorted_df.drop(columns='batch_number')
    sorted_df.to_csv(output_path, index=False)



if __name__ == "__main__":
    # audio_file_process()
    # retrospective_sis_process()
    # get_real_time_sis_without_number("/Users/taichi/Desktop/master_thesis/RealTimeSIS_without_number/", "/Users/taichi/Desktop/master_thesis/RealTimeSIS_without_number/score_only/")
    # get_real_time_sis_per_question("/Users/taichi/Desktop/master_thesis/RealTimeSIS_per_question_v1/", "/Users/taichi/Desktop/master_thesis/RealTimeSIS_per_question_v1/score_only/")
    get_summary_sis("/Users/taichi/Desktop/master_thesis/RealTimeSIS_summary_label/", "/Users/taichi/Desktop/master_thesis/RealTimeSIS_summary_label/score_only/")
    concat_summary_evaluation()
    get_real_time_sis_without_number("/Users/taichi/Desktop/master_thesis/RealTimeSIS_with_context_v2/", "/Users/taichi/Desktop/master_thesis/RealTimeSIS_with_context_v2/score_only/")
    get_real_time_sis_v2("/Users/taichi/Desktop/master_thesis/RealTimeSIS_newprompt/", "/Users/taichi/Desktop/master_thesis/RealTimeSIS_newprompt/score_only/")