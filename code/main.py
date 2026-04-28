# %%
import numpy as np
import pandas as pd

from preprocessing import *
from upload import upload_folder_to_gdrive

###
# %%
common_voice = pd.read_csv('../common-voice.csv', sep=',')
filenames = common_voice['filename']
ftest = filenames.iloc[:100]
ftest
###

### Generate mel spectograms and mfcc set
# %%
base_path = '../common-voice'
mfcc_num_output = '../dataset/mfcc_num'
mfcc_spec_output = '../dataset/mfcc_spec'
mel_num_output = '../dataset/mel_num'
mel_spec_output = '../dataset/mel_spec'

# %%
preprocess_dataset(ftest, 
                   base_path, 
                   mfcc_num_output, 
                   mfcc_spec_output, 
                   mel_num_output, 
                   mel_spec_output)

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
