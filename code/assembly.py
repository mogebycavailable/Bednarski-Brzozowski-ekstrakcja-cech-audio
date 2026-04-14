# %%
import pandas as pd
import os

df = pd.read_csv("../common-voice.csv")

mfcc_dir = "../dataset/mfcc"
mel_dir = "../dataset/mel"

mfcc_paths = []
mel_paths = []

for filename in df["filename"]:
    base = os.path.splitext(filename)[0]   # usuwa .mp3

    mfcc_paths.append(os.path.join(mfcc_dir, base + ".npy"))
    mel_paths.append(os.path.join(mel_dir, base + ".png"))

df["mfcc_path"] = mfcc_paths
df["mel_path"] = mel_paths

df.to_csv("../dataset/dataset_index.csv", index=False)

print("DONE -> dataset_index.csv")
# %%
