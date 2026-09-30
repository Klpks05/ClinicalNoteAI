import torch
from transformers import pipeline

audio_file = "data/primock57/audio/day1_consultation01_doctor.wav"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en",
    device=device
)

print("Transcribing audio...")

result = pipe(audio_file, return_timestamps=True)

print("\nWhisper transcription:")
print(result["text"])