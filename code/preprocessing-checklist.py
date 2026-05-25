
# %%
import pandas as pd
import numpy as np
import tensorflow as tf

# %%
print(tf.__version__)

# %%

df = pd.read_csv("../dataset/dataset_index_encoded.csv")

# %%
shapes_1 = []

data_size = df.shape[0]
ctr = 0

for i in range(data_size):
    num_path = df.loc[i,"mel_num_path"]
    data : np.ndarray
    data = np.load(num_path)
    shapes_1.append(data.shape[1])
    if(i % 100 == 0):
        ctr+=1
        print(str(ctr) + "/" + str(data_size/100))

shapes_1_npy = np.array(shapes_1)

# %%
print("Mediana liczby kolumn dla NPY metody MFCC wynosi "+str(np.median(shapes_1_npy)))
# MFCC Median = 336.0

# Mel Median = 

# %%
print("Min: "+str(np.min(shapes_1_npy)))
print("Max: "+str(np.max(shapes_1_npy)))
# MFCC Min Max
# Min: 32
# Max: 750

# Mel Min Max
# Min: 
# Max: 

# %%
import matplotlib.pyplot as plt

plt.boxplot(shapes_1_npy)
plt.show()

# %%
# Kod przystosowujacy ilosc probek MFCC na osi czasu do mediany wynoszacej 336

def equalize_shape(path, new_path, median):
    equalized_filepath = new_path + "\\" + path.rsplit("\\",1)[1].rsplit(".",1)[0] + "_eq.npy"
    data = np.load(path)
    d_size = data.shape[1]
    new_data : np.ndarray
    if(d_size == median):
        new_data = data
    elif(d_size > median):
        excess = d_size - median
        left = 0
        right = 0
        if(excess % 2):
            left += 1
        truncate = excess // 2
        left += truncate
        right += truncate
        if right == 0:
            new_data = data[:, left:]
        else:
            new_data = data[:, left:-right]
    else:
        missing = median - d_size
        repeats = int(np.ceil(missing / d_size))
        padding = np.tile(data, (1, repeats))
        padding = padding[:, :missing]
        new_data = np.concatenate([data, padding], axis=1)
    np.save(equalized_filepath, new_data)
    return equalized_filepath

# %%
# MFCC
sample = df.iloc[:50,:]

mfcc_ex_pth = "../dataset/mfcc_dataset_equalized_test"
mel_ex_pth = "../dataset/mel_dataset_equalized_test"

eq_list = []

for i in range(sample.shape[0]):
    pth = sample.loc[i,"mel_num_path"]
    eq_list.append(equalize_shape(pth, mel_ex_pth, 336))

for path in eq_list:
    hopefully_qualized = np.load(path)
    print(hopefully_qualized.shape)
# %%
