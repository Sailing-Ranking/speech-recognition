import whisper

model = whisper.load_model("base.en")
result = model.transcribe("data/recordings/17363ec3-6354-46fe-8e21-be68e51b537b.wav")

with open("data/recordings/17363ec3-6354-46fe-8e21-be68e51b537b.txt", "w") as file:
    file.write(result["text"])
