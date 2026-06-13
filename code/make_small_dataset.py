import pandas as pd
import numpy as np
import os

from pathlib import Path
import shutil

os.makedirs("../dataset_small", exist_ok=True)

src_root = Path("../dataset")
dst_root = Path("../dataset_small")

for directory in src_root.rglob("*"):
    if not directory.is_dir():
        continue

    rel_path = directory.relative_to(src_root)
    target_dir = dst_root / rel_path

    target_dir.mkdir(parents=True, exist_ok=True)

samples = 3000

df = pd.read_csv("../dataset/dataset_index_encoded_equalized.csv")

n = df.shape[0]

for i in range(n):
    mel_num_path = df.loc[i,"mel_eq_path"]
    mel_num_path_new = mel_num_path.replace("dataset","dataset_small")
    shutil.copy2(mel_num_path, mel_num_path_new)

    mfcc_num_path = df.loc[i,"mfcc_eq_path"]
    mfcc_num_path_new = mfcc_num_path.replace("dataset","dataset_small")
    shutil.copy2(mfcc_num_path, mfcc_num_path_new)

    mel_spec_path = df.loc[i,"mel_spec_path"]
    mel_spec_path_new = mel_spec_path.replace("dataset","dataset_small")
    shutil.copy2(mel_spec_path, mel_spec_path_new)

    mfcc_spec_path = df.loc[i,"mfcc_spec_path"]
    mfcc_spec_path_new = mfcc_spec_path.replace("dataset","dataset_small")
    shutil.copy2(mfcc_spec_path, mfcc_spec_path_new)
