# %%
import numpy as np
import pandas as pd
import os
import subprocess
import librosa
import IPython.display as ipd
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
import sys
from pathlib import Path

# %%
print(os.listdir('..'))

# %%
common_voice = pd.read_csv('../common-voice.csv', sep=',')
print(common_voice.head(10))
filenames = common_voice['filename']
filenames

# %%
base_path = '../common-voice'
mfccs_output = '../dataset/mfcc'
spects_output = '../dataset/mel'

mel_coef = pd.DataFrame()

ftest = filenames.iloc[:10]
ftest

# %%
def booster(audio):
    max_amp = np.max(np.abs(audio))
    if(max_amp==0):
        return audio
    gain = 1.0 / max_amp
    if(gain > 1.0):
        return audio * gain
    else:
        return audio

# %%
# Test typow danych
test_filename = ftest[0]

before, after = test_filename.rsplit('/', 1)
path = base_path + '/' + before + '/' + before + '/' + after
x , sr = librosa.load(path)
x = booster(x)

mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=10)

mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

print("Typ librosa.feature.mfcc: "+str(type(mfccs)))
print("Typ librosa.feature.melspectrogram: "+str(type(mel)))
print("Typ librosa.power_to_db: "+str(type(mel_db)))

display = librosa.display.specshow(mel_db, sr=sr, x_axis='time')
print("Typ librosa.display.specshow: "+str(type(display)))

print("Rozmiar mel_db: ")
print(mel_db.shape)

with np.printoptions(threshold=sys.maxsize):
    print(mel_db)

# %%
for filename in ftest:
    before, after = filename.rsplit('/', 1)
    path = base_path + '/' + before + '/' + before + '/' + after
    x , sr = librosa.load(path)
    x = booster(x)

    # MFCC:
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=10)
    mfcc_file = os.path.join(mfccs_output, after.replace('.mp3', '.npy'))
    np.save(mfcc_file, mfcc)
    
    # MEL SPECTOGRAMS:
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    spec_file = os.path.join(spects_output, after.replace('.mp3', '.png'))

    plt.figure(figsize=(15, 7))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(spec_file, bbox_inches='tight', pad_inches=0)
    plt.close()

# %%
npy_file = np.load(f"{mfccs_output}/sample-000004.npy")
print(npy_file)

# %%
# Dataset Index
for i, row in common_voice.iterrows():
    assert os.path.exists(row["mfcc_path"])
    assert os.path.exists(row["mel_path"])

# %%
RCLONE_PATH = r"D:\Programy [Studia]\rclone\rclone.exe"
local_path = "D:\Projekt magisterski\dataset"
remote_disk = "gdrive"
remote_path = "Projekt magisterski/dataset"

# %%
def upload_folder_to_gdrive(local_folder, remote_disk, remote_path):
    subprocess.run([
        RCLONE_PATH,
        "copy",
        local_folder,
        f"{remote_disk}:{remote_path}",
        "--transfers=8",
        "--checkers=16"
    ])

upload_folder_to_gdrive(local_path, remote_disk, remote_path)

# %%
pd.set_option('display.max_rows', 100)
ftest.head(100)