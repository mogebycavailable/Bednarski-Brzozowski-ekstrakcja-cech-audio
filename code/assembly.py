import pandas as pd

def assemble_record(filename, mfcc_num_path, mfcc_spec_path, mel_num_path, mel_spec_path, meta_row):
    return (
        filename,
        meta_row["age"],
        meta_row["gender"],
        meta_row["accent"],
        mfcc_num_path,
        mfcc_spec_path,
        mel_num_path,
        mel_spec_path
    )