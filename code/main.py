# %%
import numpy as np
import pandas as pd

from preprocessing import *
from upload import upload_folder_to_gdrive

import librosa

import os

###
# %%
common_voice = pd.read_csv('../common-voice.csv', sep=',')
filenames = common_voice['filename']
#ftest = filenames.iloc[:100]
#ftest
###

### Generate mel spectograms and mfcc set
# %%
base_path = '../common-voice'
mfcc_num_output = '../dataset/mfcc_num'
mfcc_spec_output = '../dataset/mfcc_spec'
mel_num_output = '../dataset/mel_num'
mel_spec_output = '../dataset/mel_spec'

# %%
'''
num_rows = len(filenames)

def delete_long_recordings(max_time_in_seconds : int):
    deleted_stat = 0
    cv = pd.read_csv('../common-voice.csv', sep=',')
    for i, filename in enumerate(filenames):
        if(i % 10000 == 0):
            print(f"Postep {i}/{num_rows}")
        before, after = filename.rsplit('/', 1)
        path = base_path + '/' + before + '/' + before + '/' + after
        time = librosa.get_duration(path=path)
        if(time > max_time_in_seconds):
            deleted_stat+=1
            csv_path = before + "/" + after
            cv = cv[cv["filename"] != csv_path]

    cv.to_csv("../common-voice.csv", index=False)
    print(f"Usunieto {deleted_stat} nagran dluzszych niz {max_time_in_seconds} sekund.")       

delete_long_recordings(8)
'''
# %%
rows = common_voice.shape[0]
process_start_range = 0
process_stop_range = common_voice.shape[0]
num_processes = rows/1000
ctr = 0

while(process_start_range<rows):
    process_stop_range = process_start_range + 1000
    if(process_stop_range >= rows):
        process_stop_range = rows - 1
    preprocess_dataset(filenames, 
                   base_path, 
                   mfcc_num_output, 
                   mfcc_spec_output, 
                   mel_num_output, 
                   mel_spec_output,
                   process_start_range,
                   process_stop_range, 
                   True)
    ctr+=1
    print(f"Wykonano {ctr}/{num_processes} procesow.")
    process_start_range+=1000

###
'''
### Upload data to Google Drive
# %%
local_path = "D:\Projekt magisterski\dataset"
remote_disk = "gdrive"
remote_path = "Projekt magisterski/dataset"

upload_folder_to_gdrive(local_path, remote_disk, remote_path)
###
# %%
'''