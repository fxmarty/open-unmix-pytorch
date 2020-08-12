import musdb
import torch
import tqdm
import librosa
import numpy as np

sampling_rate = 8192
sources_names = ['vocals','drums','bass','other']

mus = musdb.DB(
            root='../MUSDB18',
            split='train',
            subsets='train')

# Train set
print("Resampling train set to",sampling_rate,"Hz.")
print("Resampling",len(mus.tracks),"tracks.")

downsampledList = []
for track in tqdm.tqdm(mus.tracks):
    for source_name in sources_names:
        track.sources[source_name].audio = librosa.resample(track.sources[source_name].audio.T, orig_sr=44100, target_sr=sampling_rate).T
    downsampledList.append([track.sources[source_name].audio for source_name in sources_names])

np.save('trainsetDownsampled.py',downsampledList)

mus = musdb.DB(
            root='../MUSDB18',
            split='valid',
            subsets='train')

# Validation set
print("Resampling validation set to",sampling_rate,"Hz.")
print("Resampling",len(mus.tracks),"tracks.")

downsampledList = []
for track in tqdm.tqdm(mus.tracks):
    for source_name in sources_names:
        track.sources[source_name].audio = librosa.resample(track.sources[source_name].audio.T, orig_sr=44100, target_sr=sampling_rate).T
    downsampledList.append([track.sources[source_name].audio for source_name in sources_names])

np.save('validationsetDownsampled.py',downsampledList)