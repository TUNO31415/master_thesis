import torch
import torchaudio
from pyannote.audio import Pipeline
from transformers import pipeline
import os
import subprocess
from speechbox import ASRDiarizationPipeline


# https://huggingface.co/learn/audio-course/en/chapter7/transcribe-meeting

def load_api_keys():
    file_path = "token.txt"
    api_keys = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, key = line.strip().split('=')
            api_keys[name] = key
    return api_keys

hg_api_token = load_api_keys()['hg_api']

def save_pretrained_diarization():
    diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=hg_api_token
    )
    diarization_pipeline.save_model("Models/diarization")

def load_pretrained_diarization():
    diarization_pipeline = Pipeline.from_pretrained("Models/diarization")
    return diarization_pipeline

def save_pretrained_asr():
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
    )
    asr_pipeline.save_pretrained("Models/asr")


def load_pretrained_asr():
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="Models/asr", 
    )
    return asr_pipeline

def save_pretrained_asrd():
    pipeline = ASRDiarizationPipeline.from_pretrained(
        "openai/whisper-base", use_auth_token=hg_api_token
        )
    pipeline.save_pretrained("Models/asrdiarization")

def load_pretrained_asrd():
    return ASRDiarizationPipeline.from_pretrained("Models/asrdiarization")


def get_audio_from_video(input_path, output_path):
    command = f"ffmpeg -i {input_path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
    subprocess.call(command, shell=True)

def combine_audio(input_file1, input_file2, output_file):
    # FFmpeg command to concatenate two audio files
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_file1,
        '-i', input_file2,
        '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[out]',
        '-map', '[out]',
        output_file
    ]
    
    # Run FFmpeg command
    subprocess.run(ffmpeg_cmd, check=True)

def diarize_transcribe():
    output_path = "audio/conv-self-LPID_2-Batch_7-LPID_5-Batch_7-Batch_7-PID_51_NaN.mp3"
    waveform, sr = torchaudio.load(output_path)
    
    # input_tensor = torch.from_numpy(sample["audio"]["array"][None, :]).float()    
    diarization_pipeline = load_pretrained_diarization()

    outputs = diarization_pipeline(
        {"waveform" : waveform, "sample_rate" : sr}
    )

    # outputs = diarization_pipeline(
    #     {"waveform": input_tensor, "sample_rate": sample["audio"]["sampling_rate"]}
    # )

    for turn, _, speaker in outputs.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    asr_pipeline = load_pretrained_asr()

    outputs = asr_pipeline(
        input,
        generate_kwargs={"max_new_tokens": 256},
        return_timestamps=True,
    )

    print(outputs)
    return outputs

save_pretrained_asr()
save_pretrained_diarization()

asr_pipeline = load_pretrained_asr()
diarization_pipeline = load_pretrained_diarization()

# pipeline = ASRDiarizationPipeline(
#     asr_pipeline=asr_pipeline, diarization_pipeline=diarization_pipeline
# )

# # output_path = "audio/conv-self-LPID_2-Batch_7-LPID_5-Batch_7-Batch_7-PID_51_NaN.mp3"

# # output = pipeline(output_path)
