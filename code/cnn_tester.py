# %%
# NIEZBEDNE IMPORTY

import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras import layers as layer
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import Precision
from keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# %%
# WCZYTANIE DANYCH

data = pd.read_csv("../dataset/dataset_index_encoded_equalized.csv")
data

# %%
# PODZIAŁ NA ZMIENNE NIEZALEŻNE I ZALEŻNE

independents = []
dependent = []

# USUNAC REPLACE!!
for i in range(data.shape[0]):
    independents.append(np.load(data.loc[i,"mel_eq_path"].replace("dataset","dataset_small")))
    dependent.append(data.loc[i,"gender"])

X = np.array(independents)
y = np.array(dependent)

# %%
# BADANIE ZROWNOWAZENIA ZBIORU DANYCH

print(np.bincount(y.astype(int)))

# %%
# PODZIAŁ ZBIORU

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, stratify=y, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X dtype:", X_train.dtype)
print("y dtype:", y_train.dtype)

# %%
# STANDARYZACJA DANYCH DO N(0,1)
N, H, W = X_train.shape

X_train_flat = X_train.reshape(N, -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled  = scaler.transform(X_test_flat)

X_train = X_train_scaled.reshape(N, H, W)
X_test  = X_test_scaled.reshape(X_test.shape[0], H, W)

# %%
# ŁADOWANIE MODELU I CALLBACKOW

import mlp_mel_gender

model = mlp_mel_gender.load_model()
callbacks = mlp_mel_gender.load_callbacks()

model.summary()

# %%
# TRENOWANIE MODELU I ZAPIS HISTORII TRENINGU

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# %%
# WCZYTANIE NAJLEPSZEGO MODELU I PREDYKCJA

best_mlp_mfcc_model = load_model("weights/best_mel_mlp_weights.keras")

y_proba = best_mlp_mfcc_model.predict(X_test)
y_pred = (y_proba > 0.5).astype(int)

# %%
# METRYKI

cm = confusion_matrix(y_test, y_pred)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"Specificity: {specificity:.4f}")
print("F1-score :", f1_score(y_test, y_pred))

# %%
# MACIERZ POMYLEK

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Female", "Male"],
    yticklabels=["Female", "Male"]
)

plt.xlabel("Klasa przewidziana")
plt.ylabel("Klasa prawdziwa")
plt.title("Macierz pomylek - Perceptron wielowarstwowy dla Mel-spectrogramow")

plt.tight_layout()
plt.show()
# %%
