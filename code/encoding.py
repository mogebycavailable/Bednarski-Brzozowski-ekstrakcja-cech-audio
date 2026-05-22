import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('../dataset/dataset_index.csv')

print(dataset.head(5))
print(dataset.columns)
print(dataset.shape)

le_gender = LabelEncoder()
le_age = LabelEncoder()
le_accent = LabelEncoder()

dataset['gender'] = le_gender.fit_transform(dataset['gender'])
dataset['age'] = le_age.fit_transform(dataset['age'])
dataset['accent'] = le_accent.fit_transform(dataset['accent'])

print(dataset[['age','gender','accent']].head(5))


print("\nEtykiety plci (gender):")
for i,cls in enumerate(le_gender.classes_):
    print(f"\t{cls} -> {i}")

print("\nEtykiety wieku (age):")
for i,cls in enumerate(le_age.classes_):
    print(f"\t{cls} -> {i}")

print("\nEtykiety akcentu (accent):")
for i,cls in enumerate(le_accent.classes_):
    print(f"\t{cls} -> {i}")

dataset.to_csv("../dataset/dataset_index_encoded.csv")