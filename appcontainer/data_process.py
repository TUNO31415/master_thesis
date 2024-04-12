import pandas as pd
import subprocess

def get_audio_from_video(input_path, output_path):
    command = f"ffmpeg -i {input_path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
    subprocess.call(command, shell=True)

# # file_path = "https://webdata.tudelft.nl/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/LIST_unique_dyads_and_clean_videos_Manual.xlsx"
# file_path = "/Users/taichi/Downloads/LIST_unique_dyads_and_clean_videos_Manual (1).xlsx"
# df = pd.read_excel(file_path)
# new_df = df[["selfPID", "otherPID", "conv_rec_self_local", "conv_rec_other_local"]]
# csv_path = "/Users/taichi/Downloads/LIST_unique_dyads_and_clean_SELFONLY.csv"
csv_path = "/staff-umbrella/tunoMSc2023/paco_dataset/Conversation Audio/LIST_unique_dyads_and_clean_SELFONLY.csv"
# new_df.to_csv(csv_path)
# print("DONE")

df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    batch_num = row["conv_rec_self_local"].split("-")[3]
    video_1_path = f"/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/Conversations/{batch_num}/{row["conv_rec_self_local"]}.mp4"
    video_2_path = f"/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/Conversations/{batch_num}/{row["conv_rec_other_local"]}.mp4"
    audio_1_path = f"/staff-umbrella/tunoMSc2023/paco_dataset/Conversation Audio/{batch_num}/{row["conv_rec_self_local"]}.wav"
    audio_2_path = f"/staff-umbrella/tunoMSc2023/paco_dataset/Conversation Audio/{batch_num}/{row["conv_rec_other_local"]}.wav"
    get_audio_from_video(video_1_path, audio_1_path)
    get_audio_from_video(video_2_path, audio_2_path)
