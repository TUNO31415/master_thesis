import torch
from pyannote.audio import Pipeline
from datasets import load_dataset
from transformers import pipeline
import os
import subprocess


def load_api_keys():
    file_path = "token.txt"
    api_keys = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, key = line.strip().split('=')
            api_keys[name] = key
    return api_keys

hg_api_token = load_api_keys()['hg_api']

def load_pretrained_diarization():
    diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=hg_api_token
    )
    return diarization_pipeline

def get_sample_data():
    concatenated_librispeech = load_dataset(
    "sanchit-gandhi/concatenated_librispeech", split="train", streaming=True
    )
    sample = next(iter(concatenated_librispeech))
    return sample

def extract_audio_from_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of video files in input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith((".mp4", ".avi", ".mov"))]

    for video_file in video_files:
        # Construct input and output paths
        video_path = os.path.join(input_folder, video_file)
        audio_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.wav")

        # Run ffmpeg command to extract audio
        command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path]
        subprocess.run(command, capture_output=True, check=True)

sample = get_sample_data()
print(sample["audio"]["array"][None, :].shape)

input_tensor = torch.from_numpy(sample["audio"]["array"][None, :]).float()    
diarization_pipeline = load_pretrained_diarization()

outputs = diarization_pipeline(
    {"waveform": input_tensor, "sample_rate": sample["audio"]["sampling_rate"]}
)

for turn, _, speaker in outputs.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)

outputs = asr_pipeline(
    sample["audio"].copy(),
    generate_kwargs={"max_new_tokens": 256},
    return_timestamps=True,
)

print(outputs)