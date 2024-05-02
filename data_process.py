import pandas as pd
import subprocess
import os


paco_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/"
csv_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/LIST_unique_dyads_and_clean_SELFONLY.csv"
conv_data_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/curated_conv_data_combined_ffinal.csv"

def get_audio_from_video(input_path, output_path):
    command = f"ffmpeg -loglevel quiet -i {input_path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
    subprocess.call(command, shell=True)

def combine_audio(audio_1_path, audio_2_path, output_path):
    command = f"ffmpeg -loglevel quiet -i {audio_1_path} -i {audio_2_path} -filter_complex amix=inputs=2:duration=longest {output_path}"
    subprocess.call(command, shell=True)

def audio_file_process():
    df = pd.read_csv(csv_path)
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

    c1 = row['conv_SP_SIS_C3']
    c2 = row['conv_SP_SIS_C4r']
    fd1 = row['conv_SP_SIS_FD5']
    fd2 = row['conv_SP_SIS_FD6r']
    ic1 = row['conv_SP_SIS_IC7']
    ic2 = row['conv_SP_SIS_IC8r']
    md1 = row['conv_SP_SIS_MD1']
    md2 = row['conv_SP_SIS_MD2r']
    p1 = row['conv_SP_SIS_P10r']
    p2 = row['conv_SP_SIS_P9']

    return {'BatchNum' : batch, 'selfPID' : self, 'otherPID' : other, 'MD' : '', 'CI' : '', 'FI' : '','IC' : '', 'P' : ''}

def retrospective_sis_process():
    audio_file_df = pd.read_csv(csv_path)
    audio_file_df = audio_file_df.dropna()
    audio_file_df = audio_file_df.drop_duplicates()
    conv_data_df = pd.read_csv(conv_data_path)
    output_list = []

    for index, row in audio_file_df.iterrows():
        speaker0_name = row["selfPID"]
        speaker1_name = row["otherPID"]
        speaker0_LPID = row["conv_rec_self_local"].split("-")[2]
        speaker1_LPID = row["conv_rec_other_local"].split("-")[2]
        batch_num = row["conv_rec_self_local"].split("-")[3]

        transcription_path = paco_path + f"ConversationAudio/transcription/{batch_num}_{speaker0_name}_{speaker1_name}.csv"

        if not os.path.exists(transcription_path):
            continue
        
        matching_row_0 = conv_data_df[(conv_data_df['globalPID'] == speaker0_name) & (conv_data_df['batchID'] == batch_num) &  (conv_data_df['conv_otherLPID'] == speaker1_LPID)]
        matching_row_1 = conv_data_df[(conv_data_df['globalPID'] == speaker1_name) & (conv_data_df['batchID'] == batch_num) &  (conv_data_df['conv_otherLPID'] == speaker0_LPID)]

        retro_sis_0 = creat_df_entry(batch_num, speaker0_name, speaker1_name, matching_row_0)
        retro_sis_1 = creat_df_entry(batch_num, speaker1_name, speaker0_name, matching_row_1)

        output_list.append(retro_sis_0)
        output_list.append(retro_sis_1)
        print(f"DONE --- {index} --- {batch_num} {speaker0_name}+{speaker1_name}")
    
    output_csv_path = paco_path + "retrospective_sis.csv"
    df = pd.DataFrame(output_list)
    df.to_csv(output_csv_path)

    print("DONE")

def process_real_time_sis():
    # real_time_sis_path = paco_path + "RealTimeSIS/"
    # output_folder_path = paco_path + "RealTimeSIS/score_only/"

    real_time_sis_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v1"
    output_folder_path = "/Users/taichi/Desktop/master_thesis/RealTimeSIS_v1/score_only"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for file in os.listdir(real_time_sis_path):
        df = pd.read_csv(real_time_sis_path + file)
        rows = []
        ouput_path = output_folder_path + f"score_only_{file}"

        if os.path.exists(ouput_path):
            continue

        for index, row in df.iterrows():
            text = row['0']
            part = text.split("[/INST]", 1)[1]
            numbers_list = list(filter(str.isdigit, part))
            scores =  [int(num) for num in numbers_list]
            scores = scores[-5:]
            row = {'index' : index, 'MD' : scores[0], 'CI' : scores[1], 'FI' : scores[2],'IC' : scores[3], 'P' : scores[4]}
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(ouput_path)
        print(f"DONE --- score_only_{file} SAVED --- ")

    print("DONE")

    # real_time_sis_path = "/Users/taichi/Desktop/master_thesis/real-time SIS test/"
    # output_folder_path = "/Users/taichi/Desktop/master_thesis/real-time SIS test/scoreonly/"
    # file = "output_SIS_00_ONLYNUM.csv"

    # if not os.path.exists(output_folder_path):
    #     os.makedirs(output_folder_path)

    # df = pd.read_csv(real_time_sis_path + file)
    # rows = []
    # ouput_path = output_folder_path + f"score_only_{file}"
    # for index, row in df.iterrows():
    #     text = row['0']
    #     part = text.split("[/INST]", 1)[1]
    #     numbers_list = list(filter(str.isdigit, part))
    #     scores =  [int(num) for num in numbers_list]
    #     scores = scores[-5:]
    #     row = {'index' : index, 'MD' : scores[0], 'CI' : scores[1], 'FI' : scores[2],'IC' : scores[3], 'P' : scores[4]}
    #     rows.append(row)

    # df = pd.DataFrame(rows)
    # df.to_csv(ouput_path)
    
    # print("DONE")

if __name__ == "__main__":
    # audio_file_process()
    # retrospective_sis_process()
    process_real_time_sis()