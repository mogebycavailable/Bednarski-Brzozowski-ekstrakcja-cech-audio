import pandas as pd

def assemble_record(filename, mfcc_path, mel_path, meta_row):
    return (
        filename,
        meta_row["age"],
        meta_row["gender"],
        meta_row["accent"],
        mfcc_path,
        mel_path
    )

'''
filename: cv-valid-dev/sample-000030.mp3
mfcc_file: ../dataset/mfcc\sample-000030.npy
spec_file: ../dataset/mel\sample-000030.png
dataset_assembly is working...
'''