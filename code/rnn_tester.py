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
import tensorflow as tf
from keras.layers import Normalization

# %%
# WCZYTANIE DANYCH

df = pd.read_csv("../dataset/dataset_index_encoded_equalized.csv")
df = df.iloc[:6000,:]
df

# %%
# PRZEJSCIE NA TYP NUMPY.NDARRAY
data = df[["mel_eq_path", "accent"]].to_numpy()
data

# %%
# BADANIE ZROWNOWAZENIA ZBIORU
print(np.bincount(data[:,1].astype(int)))

# %%
# PODZIAŁ NA ZBIÓR TRENINGOWY, TESTOWY I WALIDACYJNY

X = data[:,0]
y = data[:,1].astype(np.int32)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# %%

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.60,
    random_state=42,
    stratify=y_temp
)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

# %%
# WERYFIKACJA TYPOW DANYCH
print(type(X_train[0]))
print(X_train[0])

# %%
# USTALENIE WAG DLA NIEZRÓWNOWAŻONEGO ZBIORU DANYCH
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print(np.unique(y_train))
print(class_weights)
print(X_train.shape)

# %%
# PRZEJŚCIE NA TYP DATASET
def load_numpy(path):
    file = np.load(path.decode())
    return file.astype(np.float32)

# DEFINICJA FUNKCJI MAPUJACEJ
def load(path : str, label):
    #path = path.replace("dataset","dataset_small")
    tensor = tf.numpy_function(load_numpy, [path], tf.float32)
    tensor.set_shape((256, 336))
    tensor = tf.transpose(tensor)
    return tensor, label

# %%
# TEST FUNKCJI MAPUJACEJ
'''
tensor, label = load(X_train[0], y_train[0])

print(tensor.shape)
print(label)
'''

# %%
# ŁADOWANIE TENSORÓW RZĘDU DRUGIEGO (MACIERZY/TABLIC) W LOCIE PODCZAS TRENINGU
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000)
    .map(load, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(load, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(load, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

# %%
# STANDARYZACJA TF.DATA.DATASET
normalizer = Normalization(axis=-1)
normalizer.adapt(train_dataset.map(lambda x, y: x))
train_dataset = train_dataset.map(lambda x, y: (normalizer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalizer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalizer(x), y))

# %%
# ŁADOWANIE MODELU I CALLBACKOW

import rnn_mel_accent

model = rnn_mel_accent.load_model()
callbacks = rnn_mel_accent.load_callbacks()

model.summary()

# %%
# TRENOWANIE MODELU I ZAPIS HISTORII TRENINGU

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# %%
# WCZYTANIE NAJLEPSZEGO MODELU I PREDYKCJA

best_rnn_model = load_model("weights/best_mel_rnn_accent_model.keras")

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
