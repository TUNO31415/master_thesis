import pandas as pd
import subprocess
import os
import tqdm

def get_audio_from_video(input_path, output_path):
    command = f"ffmpeg -loglevel quiet -i {input_path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
    subprocess.call(command, shell=True)

def combine_audio(audio_1_path, audio_2_path, output_path):
    command = f"ffmpeg -loglevel quiet -i {audio_1_path} -i {audio_2_path} -filter_complex amix=inputs=2:duration=longest {output_path}"
    subprocess.call(command, shell=True)

# # file_path = "https://webdata.tudelft.nl/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/LIST_unique_dyads_and_clean_videos_Manual.xlsx"
# file_path = "/Users/taichi/Downloads/LIST_unique_dyads_and_clean_videos_Manual (1).xlsx"
# df = pd.read_excel(file_path)
# new_df = df[["selfPID", "otherPID", "conv_rec_self_local", "conv_rec_other_local"]]
# csv_path = "/Users/taichi/Downloads/LIST_unique_dyads_and_clean_SELFONLY.csv"
csv_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/LIST_unique_dyads_and_clean_SELFONLY.csv"
# new_df.to_csv(csv_path)
# print("DONE")

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
    