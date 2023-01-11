import noisereduce as nr
import torch
import torchaudio

# load data
y, sr = torchaudio.load("data/recordings/17363ec3-6354-46fe-8e21-be68e51b537b.wav")

waveform = torch.tensor(nr.reduce_noise(y=y, y_noise=y, sr=sr))

torchaudio.save("test.wav", src=waveform, sample_rate=sr)
