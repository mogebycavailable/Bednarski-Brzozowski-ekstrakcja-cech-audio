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
mfccs_output = '../dataset/mfcc'
spects_output = '../dataset/mel'

# %%
#test_datatypes(ftest, base_path)
show_mfcc_output(mfccs_output, '000013')
show_mel_spec_output(spects_output, '000005')

# %%
preprocess_dataset(ftest, base_path, mfccs_output, spects_output)

###

### Upload data to Google Drive
# %%
local_path = "D:\Projekt magisterski\dataset"
remote_disk = "gdrive"
remote_path = "Projekt magisterski/dataset"

upload_folder_to_gdrive(local_path, remote_disk, remote_path)
###
# %%
