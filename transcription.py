import torch
from pyannote.audio import Pipeline
from datasets import load_dataset
from transformers import pipeline


def load_api_keys():
    file_path = "token.txt"
    api_keys = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, key = line.strip().split('=')
            api_keys[name] = key
    return api_keys

hg_api_token = load_api_keys()['hg_api']

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=hg_api_token
)

concatenated_librispeech = load_dataset(
    "sanchit-gandhi/concatenated_librispeech", split="train", streaming=True
)
sample = next(iter(concatenated_librispeech))

input_tensor = torch.from_numpy(sample["audio"]["array"][None, :]).float()
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