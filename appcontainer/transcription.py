!pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
!pip install git+https://github.com/m-bain/whisperx.git

import whisperx
import gc 
import csv
import os
import pandas as pd
import torch
from transformers import pipeline
from huggingface_hub import login

def get_video_names():
    file_path = "https://webdata.tudelft.nl/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/LIST_unique_dyads_and_clean_videos_Manual.xlsx"
    df = pd.read_excel(file_path)
    new_df = df[["selfPID", "otherPID", "conv_rec_self_local", "conv_rec_other_local"]]
    csv_path = "https://webdata.tudelft.nl/staff-umbrella/tunoMSc2023/paco_dataset/Conversation/LIST_unique_dyads_and_clean_self_videos.csv"
    new_df.to_csv(csv_path)
    print("DONE")

def audio_from_video():
    video_file_path = "https://webdata.tudelft.nl/staff-umbrella/tunoMSc2023/"

def main():
    device = "cuda" 
    audio_file = "/kaggle/input/sample2/sample_dyad.wav"
    audio_file = "https://webdata.tudelft.nl/staff-umbrella/tunoMSc2023/"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    # print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"]) # segments are now assigned speaker IDs