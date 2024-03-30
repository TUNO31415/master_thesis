import torch
from transformers import pipeline
import subprocess
import csv

def save_pretrained_asr():
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
    )
    asr_pipeline.save_pretrained("Models/asr")
    print("Saved")

def load_pretrained_asr():
    p = pipeline("automatic-speech-recognition", model="Models/asr", tokenizer="Models/asr")
    print("loaded!")
    return p


def transcribe(input, asr_pipeline):
    outputs = asr_pipeline(
        input,
        generate_kwargs={"max_new_tokens": 256},
        chunk_length_s = 200,
        return_timestamps=True,
    )
    return outputs["chunks"]

def get_audio_from_video(input_path, output_path):
    command = f"ffmpeg -i {input_path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
    subprocess.call(command, shell=True)

def create_csv(data, filename):
    # Extracting column names from the keys of the first dictionary
    fields = data[0].keys()

    # Writing data to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        
        # Writing header
        writer.writeheader()
        
        # Writing rows
        writer.writerows(data)

# save_pretrained_asr()
asr_pipeline = load_pretrained_asr()


sample_video_path = "/Users/taichi/Downloads/conv-self-LPID_2-Batch_7-LPID_5-Batch_7-Batch_7-PID_51_NaN.mp4"
output_path = "audio/conv-self-LPID_2-Batch_7-LPID_5-Batch_7-Batch_7-PID_51_NaN.mp3"

# get_audio_from_video(sample_video_path, output_path)

output = transcribe(output_path, asr_pipeline)
create_csv(output, "sample.csv")

