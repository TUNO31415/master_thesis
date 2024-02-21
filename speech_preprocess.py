import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# https://huggingface.co/openai/whisper-large-v3

save_path = "./models/whisper"

def save_whisper():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)

def load_whisper():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(save_path)
    processor = AutoProcessor.from_pretrained(save_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe

# save_whisper()
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

pipe = load_whisper()
result = pipe(sample)
print(result["text"])
