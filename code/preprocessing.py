# %%
import numpy as np
import pandas as pd
import os
import librosa
import IPython.display as ipd
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display

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
pd.set_option('display.max_rows', 100)
ftest.head(100)

# %%
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

# %%
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

# %%
df = pd.DataFrame(mfccs)
df.to_csv('/kaggle/working/mfccs.csv', index=False)

# %%
print('Dane do CNN')

# %%
