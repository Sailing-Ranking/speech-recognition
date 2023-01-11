import wave
from uuid import uuid4

import librosa
import noisereduce as nr
import pyaudio
import whisper

model = whisper.load_model("base.en")

chunk = 1024  # record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
sample_rate = 44100  # record at 44100 samples per second
seconds = 5
filename = f"data/recordings/{uuid4()}"

p = pyaudio.PyAudio()  # create an interface to PortAudio

print("Recording")

stream = p.open(
    format=sample_format,
    channels=channels,
    rate=sample_rate,
    frames_per_buffer=chunk,
    input=True,
)

frames = []  # initialize array to store frames

# store data in chunks for 5 seconds
for i in range(0, int(sample_rate / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# stop and close the stream
stream.stop_stream()
stream.close()
# terminate the PortAudio interface
p.terminate()

print("Finished recording")

# save the recorded data as a WAV file
wf = wave.open(f"{filename}.wav", "wb")
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(sample_rate)
wf.writeframes(b"".join(frames))
wf.close()

# load the audio file
waveform, sr = librosa.load(f"{filename}.wav")

# reduce the background noise
waveform = nr.reduce_noise(y=waveform, sr=sr)

# perform speech recognition
result = model.transcribe(waveform, fp16=False)

# save transcription
with open(f"{filename}.txt", "w") as file:
    print(result["text"])
    file.write(result["text"])
