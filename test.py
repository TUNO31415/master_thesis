import pandas as pd
from gpt_utils import split_files_into_chunks


def main():
    chunks = split_files_into_chunks("/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/ConversationAudio/transcription/", 8)
    print(chunks[3])

if __name__ == "__main__":
    main()