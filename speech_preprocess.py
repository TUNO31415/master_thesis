from pyannote.audio import Pipeline
import main

# https://huggingface.co/pyannote/speaker-diarization
def diarization(file_path, output_path):
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=main.get_api_key("hg_api_token"))
    # apply the pipeline to an audio file
    diarization = pipeline(file_path, num_speakers=2)

    # dump the diarization output to disk using RTTM format
    with open(output_path, "w") as rttm:
        diarization.write_rttm(rttm)