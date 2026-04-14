# %%
import pandas as pd
import os

# %%
df = pd.read_csv("../dataset/dataset_index.csv")

missing = []

# %%
for i, row in df.iterrows():
    if not os.path.exists(row["mfcc_path"]):
        missing.append(("MFCC", row["filename"], row["mfcc_path"]))
    
    if not os.path.exists(row["mel_path"]):
        missing.append(("MEL", row["filename"], row["mel_path"]))

print("Braki:", len(missing))
print(missing[:10])
# %%
