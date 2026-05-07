# %%
import numpy as np
import pandas as pd

from preprocessing import *
from upload import upload_folder_to_gdrive

import librosa

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
sampling_rates = []
num_rows = len(filenames)

for i, filename in enumerate(filenames):
    if(i % 10000 == 0):
        print(f"Postep {i}/{num_rows}")
    before, after = filename.rsplit('/', 1)
    path = base_path + '/' + before + '/' + before + '/' + after
    sr = librosa.get_samplerate(path)
    if(sr not in sampling_rates):
        sampling_rates.append(sr)
    
    
    
print("Liczba roznych czestotliwosci probkowania to "+str(len(sampling_rates)))
print("A oto one: ")
print(sampling_rates)

# %%
rows = common_voice.shape[0]
name_duplicates = 0
used_names = []

for i in range(rows):
    before, after = common_voice.iloc[i,0].rsplit('/', 1)
    if(after in used_names):
        name_duplicates+=1
    else:
        used_names.append(after)

print("Wykryto "+str(name_duplicates)+" duplikatow nazw.")


# %%
preprocess_dataset(filenames, 
                   base_path, 
                   mfcc_num_output, 
                   mfcc_spec_output, 
                   mel_num_output, 
                   mel_spec_output,
                   True)

# %%
#test_datatypes(ftest, base_path)
show_num_output(mfcc_num_output, '000004')
show_spec_output(mfcc_spec_output)
show_num_output(mel_num_output)
show_spec_output(mel_spec_output, '000004')

###

### Upload data to Google Drive
# %%
local_path = "D:\Projekt magisterski\dataset"
remote_disk = "gdrive"
remote_path = "Projekt magisterski/dataset"

upload_folder_to_gdrive(local_path, remote_disk, remote_path)
###
# %%
