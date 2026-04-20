import numpy as np
import pandas as pd
import os
import librosa
from IPython.display import Image, display
import matplotlib.pyplot as plt
import librosa.display
import sys
from pathlib import Path

from audio_booster import booster
from assembly import assemble_record

dataset = pd.read_csv('../common-voice.csv', sep=',').set_index('filename')
records = []

def preprocess_dataset(df, base_path, mfccs_output, spects_output):
    for filename in df:
        if filename not in dataset.index:
            continue

        meta_row = dataset.loc[filename]

        before, after = filename.rsplit('/', 1)
        path = base_path + '/' + before + '/' + before + '/' + after
        x , sr = librosa.load(path)
        x = booster(x)

        # MFCC:
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=10)
        #mfcc_path = os.path.join(mfccs_output, after.replace('.mp3', '.npy'))
        mfcc_path = Path(mfccs_output) / (Path(after).stem + ".npy")
        np.save(mfcc_path, mfcc)
        
        # MEL SPECTOGRAMS:
        mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        #mel_path = os.path.join(spects_output, after.replace('.mp3', '.png'))
        mel_path = Path(spects_output) / (Path(after).stem + ".png")

        plt.figure(figsize=(15, 7))
        librosa.display.specshow(mel_db, sr=sr, x_axis='time')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(mel_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Assembling
        records.append(assemble_record(
            filename,
            str(mfcc_path),
            str(mel_path),
            meta_row
        ))
    
    final_df = pd.DataFrame(records, columns=[
        "filename",
        "age",
        "gender",
        "accent",
        "mfcc_path",
        "mel_path"
    ])
    final_df.to_csv("../dataset/dataset_index.csv", index=False)

### Additional functions
def test_datatypes(df, base_path):
    test_filename = df[0]

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

def show_mfcc_output(path, id='000004'):
    try:
        npy_file = np.load(f"{path}/sample-{id}.npy")
        print(npy_file)

        plt.figure(figsize=(15, 7))
        librosa.display.specshow(npy_file, x_axis='time')
        plt.colorbar()
        plt.title('MFCC Spectrogram')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("File not found:", path)

def show_mel_spec_output(path, id='000004'):
    try:
        display(Image(f"{path}/sample-{id}.png"))
    except FileNotFoundError:
        print("File not found:", path)