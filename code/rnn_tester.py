# %%
# NIEZBEDNE IMPORTY

import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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

for i in range(data.shape[0]):
    independents.append(np.load(data.loc[i,"mfcc_eq_path"].replace("dataset","dataset_small")))
    dependent.append(int(data.loc[i,"gender"]))

X = np.array(independents)
X = np.transpose(X, (0, 2, 1))
y = np.array(dependent)

print("X.shape = "+str(X.shape))
print("y.shape = "+str(y.shape))

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

# USTALENIE WAG DLA NIEZRÓWNOWAŻONEGO ZBIORU DANYCH
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))

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

import rnn_mfcc_gender

model = rnn_mfcc_gender.load_model()
callbacks = rnn_mfcc_gender.load_callbacks()

model.summary()

# %%
# TRENOWANIE MODELU I ZAPIS HISTORII TRENINGU

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# %%
# WCZYTANIE NAJLEPSZEGO MODELU I PREDYKCJA

best_rnn_model = load_model("weights/best_mfcc_rnn_model.keras")

y_proba = best_rnn_model.predict(X_test)
y_pred = (y_proba > 0.5).astype(int)

# %%
# METRYKI

y_test = np.concatenate([
    y.numpy().reshape(-1)
    for _, y in y_test
])

# %%
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
plt.title("Macierz pomylek - Sieć rekurencyjna dla współczynników MFCC")

plt.tight_layout()
plt.show()
# %%
