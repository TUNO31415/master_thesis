import pandas as pd

file_path = "https://webdata.tudelft.nl/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/LIST_unique_dyads_and_clean_videos_Manual.xlsx"
df = pd.read_excel(file_path)
new_df = df[["selfPID", "otherPID", "conv_rec_self_local", "conv_rec_other_local"]]
csv_path = "https://webdata.tudelft.nl/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/LIST_unique_dyads_and_clean_self_videos.csv"
new_df.to_csv(csv_path)
print("DONE")