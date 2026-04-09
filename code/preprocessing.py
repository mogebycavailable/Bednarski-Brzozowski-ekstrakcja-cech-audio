# %%
import numpy as np
import pandas as pd
import os
import librosa
import IPython.display as ipd
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
import sys
from upload import upload_file_to_google_drive
from pathlib import Path

# %%
print(os.listdir('..'))

# %%
filenames = pd.read_csv('../common-voice.csv', sep=',')['filename']
filenames

# %%
# Nie odpalać chuzia na józia bo to zajmuje dłuuugo i pije pamięć RAM jak żul tanie wino
base_path = '../common-voice'
base_output = '../spectograms'

mel_coef = pd.DataFrame()

ftest = filenames.iloc[:100]
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

    #mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=10)

    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    out_file = os.path.join(base_output, after.replace('.mp3', '.png'))

    plt.figure(figsize=(15, 7))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.close()

# %%
sciezka = Path(base_output)

max_files = sum(1 for f in sciezka.rglob("*") if f.is_file())

print(f"Uploaduje {max_files} plikow...")
print("Postep: ")
upload_file_to_google_drive()

# %%
pd.set_option('display.max_rows', 100)
ftest.head(100)