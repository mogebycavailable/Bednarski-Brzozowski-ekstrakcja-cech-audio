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
from PIL import Image

from tf_train_test_split import train_val_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# %%
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# %%
# WCZYTANIE DANYCH

df = pd.read_csv("../dataset_small/dataset_index_encoded_equalized.csv")
df

# %%
# PRZEJSCIE NA TYP NUMPY.NDARRAY
data = df[["mel_spec_path", "accent"]].to_numpy()
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
# PRZEJŚCIE NA TYP DATASET

# DEFINICJA FUNKCJI MAPUJACEJ
def load(path : str, label):
    path = path.replace("dataset","dataset_small")
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.resize(img, (128, 256))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# %%
# TEST FUNKCJI MAPUJACEJ

img, label = load(X_train[0], y_train[0])

print(img.shape)
print(label)

# %%
# ŁADOWANIE ZDJĘĆ W LOCIE PODCZAS TRENINGU
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000)
    .map(load, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(load, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(load, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)


# %%
# ŁADOWANIE MODELU I CALLBACKOW

import cnn_mel_accent

model = cnn_mel_accent.load_model()
callbacks = cnn_mel_accent.load_callbacks()

model.summary()

# %%
# TRENOWANIE MODELU I ZAPIS HISTORII TRENINGU

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# %%
# WCZYTANIE NAJLEPSZEGO MODELU I PREDYKCJA

best_mlp_mel_model = load_model("weights/best_mfcc_cnn_model.keras")

y_proba = best_mlp_mel_model.predict(test_ds)
y_pred = (y_proba > 0.5).astype(int)

# %%
# METRYKI

y_test = np.concatenate([
    y.numpy().reshape(-1)
    for _, y in test_ds
])

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
plt.title("Macierz pomylek - Sieć konwolucyjna dla MEL-spectrogramow")

plt.tight_layout()
plt.show()
# %%

X_np = np.stack([
    x.numpy()
    for x, _ in ds
])

print(X_np.min())
print(X_np.max())
print(X_np.mean())
print(X_np.std())

print(X_np.shape)
print(X_np.dtype)


# %%
np.isnan(X_np).sum()

# %%
np.isinf(X_np).sum()
# %%
