import wave
from uuid import uuid4

import pyaudio

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
sample_rate = 44100  # Record at 44100 samples per second
seconds = 5
filename = f"data/recordings/{uuid4()}.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print("Recording")

stream = p.open(
    format=sample_format,
    channels=channels,
    rate=sample_rate,
    frames_per_buffer=chunk,
    input=True,
)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(sample_rate / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print("Finished recording")

# Save the recorded data as a WAV file
wf = wave.open(filename, "wb")
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(sample_rate)
wf.writeframes(b"".join(frames))
wf.close()
