import whisperx
import os
import pandas as pd
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# paco_path = "/Volumes/SFTP/staff-umbrella/tunoMSc2023/paco_dataset/"
paco_path = "/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/"
def main():
    device = "cuda"
    # device = "cpu"
    csv_file_path = paco_path + "ConversationAudio/LIST_unique_dyads_and_clean_SELFONLY.csv"
    df = pd.read_csv(csv_file_path)
    df = df.dropna()
    df = df.drop_duplicates()

    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    # batch_size = 1 # reduce if low on GPU mem
    # compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="en")
    print("MODEL SUCCESSFULLY LOADED1")
    # model = whisperx.load_model("base.en", device, compute_type=compute_type, language="en")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
    print("MODEL SUCCESSFULLY LOADED2")
    audio_folder_path = paco_path + "ConversationAudio/"
    output_folder_path = paco_path + "ConversationAudio/transcription/"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for index, row in df.iterrows():

        batch_num = row["conv_rec_self_local"].split("-")[3]
        speaker0 = row["conv_rec_self_local"]
        speaker1 = row["conv_rec_other_local"]
        speaker0_name = row["selfPID"]
        speaker1_name = row["otherPID"]
        audio_path_0 = audio_folder_path + f"{batch_num}/{speaker0}.wav"
        audio_path_1 = audio_folder_path + f"{batch_num}/{speaker1}.wav"
        audio_path_combined = paco_path + f"ConversationAudio/Combined/{speaker0_name}_{speaker1_name}.wav"

        if os.path.exists(output_folder_path + f"{batch_num}_{speaker0_name}_{speaker1_name}.csv"):
            continue
        
        if not os.path.exists(audio_path_combined):
            continue

        audio = whisperx.load_audio(audio_path_combined)
        result = model.transcribe(audio, batch_size=batch_size)
        # print(result["segments"]) # before alignment

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache();

        # 2. Align whisper output
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); 

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        output_df = pd.DataFrame(data=result["segments"])
        mask = output_df["speaker"].isin(["SPEAKER_00"])
        speaker0_dir_text = output_df[mask]["text"]

        audio = whisperx.load_audio(audio_path_0)
        result = model.transcribe(audio, batch_size=batch_size)

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache();

        # 2. Align whisper output
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); 
        
        speaker0_df = pd.DataFrame(data=result["segments"])
        speaker0_ind_text = speaker0_df["text"]

        similarity_match = len(set(speaker0_dir_text).intersection(set(speaker0_ind_text))) / len(speaker0_ind_text)

        if similarity_match > 0.8:
            output_df["speaker"] = output_df["speaker"].replace("SPEAKER_00", speaker0_name)
            output_df["speaker"] = output_df["speaker"].replace("SPEAKER_01", speaker1_name)
        else:
            output_df["speaker"] = output_df["speaker"].replace("SPEAKER_00", speaker1_name)
            output_df["speaker"] = output_df["speaker"].replace("SPEAKER_01", speaker0_name)
        
        output_df.to_csv(output_folder_path + f"{batch_num}_{speaker0_name}_{speaker1_name}.csv")

        print(f"{batch_num}_{speaker0_name}_{speaker1_name}.csv SAVED")
        print(f"{index} / {len(df)}")

    print("DONE")
    
if __name__ == "__main__":
    main()