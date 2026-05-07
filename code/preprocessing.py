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

MEL_WIDTH = 512
MEL_HEIGHT = 256

MFCC_WIDTH = 256
MFCC_HEIGHT = 128

N_MELS = 256
N_MFCC = 20

DPI=128

def preprocess_dataset(df, base_path, mfcc_num_output, mfcc_spec_output, mel_num_output, mel_spec_output, is_grayscale=False):
    duplicate_counter = 0

    mel_figsize = (MEL_WIDTH/DPI, MEL_HEIGHT/DPI)
    mfcc_figsize = (MFCC_WIDTH/DPI, MFCC_HEIGHT/DPI)
    size = df.shape[0]
    iterator = 0

    dictionary = {}

    for filename in df:
        if filename not in dataset.index:
            continue

        meta_row = dataset.loc[filename]

        before, after = filename.rsplit('/', 1)
        path = base_path + '/' + before + '/' + before + '/' + after

        x , sr = librosa.load(path, sr=48000)
        x = booster(x)

        if(after in dictionary):
            dictionary[after] +=1
            duplicate_counter+=1
            before_dot, after_dot = after.rsplit('.',1)
            before_dot += f"-{dictionary[after]}"
            after = before_dot +"."+ after_dot
        else:
            dictionary[after] = 0

        # MFCC numeric:
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=N_MFCC)
        mfcc_num_path = Path(mfcc_num_output) / (Path(after).stem + ".npy")
        np.save(mfcc_num_path, mfcc)

        # MFCC spectograms:
        mfcc_spec_path = Path(mfcc_spec_output) / (Path(after).stem + ".png")
        fig = plt.figure(figsize=mfcc_figsize, dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        if(is_grayscale):
            librosa.display.specshow(mfcc, sr=sr, cmap="gray")
        else:
            librosa.display.specshow(mfcc, sr=sr)
        ax.set_axis_off()
        plt.savefig(mfcc_spec_path, pad_inches=0, dpi=DPI)
        plt.close()


        # MEL numeric:
        mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_num_path = Path(mel_num_output) / (Path(after).stem + ".npy")
        np.save(mel_num_path, mel_db)

        # MEL spectograms:
        mel_spec_path = Path(mel_spec_output) / (Path(after).stem + ".png")

        fig = plt.figure(figsize=mel_figsize, dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        if(is_grayscale):
            librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='m', cmap="gray")
        else:
            librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='m')
        ax.set_axis_off()
        plt.savefig(mel_spec_path, pad_inches=0, dpi=DPI)
        plt.close()
        
        # Assembling
        records.append(assemble_record(
            filename,
            str(mfcc_num_path),
            str(mfcc_spec_path),
            str(mel_num_path),
            str(mel_spec_path),
            meta_row
        ))

        iterator+=1
        if(iterator%100==0):
            print("Gotowe "+str(iterator*100/float(size))+"%")
    
    final_df = pd.DataFrame(records, columns=[
        "filename",
        "age",
        "gender",
        "accent",
        "mfcc_num_path",
        "mfcc_spec_path",
        "mel_num_path",
        "mel_spec_path"
    ])
    final_df.to_csv("../dataset/dataset_index.csv", index=False)

    print(f"Uratowano {duplicate_counter} duplikatow nazw.")

### Additional functions
def test_datatypes(df, base_path):
    test_filename = df[0]

    before, after = test_filename.rsplit('/', 1)
    path = base_path + '/' + before + '/' + before + '/' + after
    x , sr = librosa.load(path)
    x = booster(x)

    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=10)

    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=256)
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

def show_num_output(path, id='000004'):
    try:
        npy_file = np.load(f"{path}/sample-{id}.npy")
        print(npy_file)
        print(npy_file.shape)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(npy_file, x_axis='time')
        plt.colorbar()
        plt.title('MFCC Spectrogram')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("File not found:", path)

def show_spec_output(path, id='000004'):
    try:
        display(Image(f"{path}/sample-{id}.png"))
    except FileNotFoundError:
        print("File not found:", path)