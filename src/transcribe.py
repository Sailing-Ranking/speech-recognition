import librosa
import whisper

model = whisper.load_model("base.en")

waveform, sr = librosa.load("data/recordings/a3e6d6ab-9217-48ad-8982-67648f0ca0f0.wav")

result = model.transcribe(waveform, fp16=False)

with open("data/transcriptions/a3e6d6ab-9217-48ad-8982-67648f0ca0f0.txt", "w") as file:
    file.write(result["text"])
