import whisperx
import gc 
import csv

device = "cpu" 
audio_file = "speaker1.wav"
batch_size = 1 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("small", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language="en")
# print(result["segments"]) # before alignment

print("Transcription DONE")

# # delete model if low on GPU resources
# # import gc; gc.collect(); torch.cuda.empty_cache(); del model

# # 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print("Alignment DONE")

print(result["segments"]) # after alignment

# Specify the field names (column headers)
# fieldnames = ["text", "start", "end"]

# Open the CSV file in write mode
# with open("output.csv", 'w', newline='') as file:
#     # Create a CSV DictWriter object
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
    
#     # Write the header row
#     writer.writeheader()
    
#     # Write each segment as a row in the CSV file
#     for segment in result["segments"]:
#         writer.writerow(segment)

print('Process done')