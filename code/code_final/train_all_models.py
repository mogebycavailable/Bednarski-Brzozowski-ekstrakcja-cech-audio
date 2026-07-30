# %%
# GLOBALNE USTAWIENIA

INCLUDE_INVALID = True # True, jezeli zbior invalid ma byc uwzgledniony, False jezeli ma byc nieuwzgledniony

BATCH_SIZE = 32 # Rozmiar wsadu

N_ROWS = -1 # -1 jezeli ma byc caly dataset, podac ilosc jezeli chcemy wziac pod uwage tylko pierwszych N wierszy zbioru danych

RANDOM_STATE = 42 # None, jezeli chcemy prawdziwie losowy podzial

VERBOSE = 1 # USTAWIC NA 1, JEZELI CHCEMY WYSWIETLAC DODATKOWE INFORMACJE, 0 JEZELI NIE CHCEMY    

# %%
# NIEZBEDNE IMPORTY

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

import models
from models_config import MODELS

from keras.layers import Normalization
from keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# %%
# DEFINICJA FUNKCJI MAPUJACYCH I FUNKCJI WSPIERAJACYCH FUNKCJE MAPUJACE

# FUNKCJA WSPIERAJACA FUNKCJE MAPUJACA DLA PLIKOW .numpy
def fetch_numpy(path):
    file = np.load(path.decode())
    return file.astype(np.float32)

# DEFINICJA FUNKCJI MAPUJACEJ DLA PLIKOW .numpy
def load_numpy(path : str, label):
    tensor = tf.numpy_function(fetch_numpy, [path], tf.float32)
    tensor.set_shape((256, 336))
    tensor = tf.transpose(tensor)
    return tensor, label

# DEFINICJA FUNKCJI MAPUJACEJ DLA OBRAZOW .png
def load_img(path : str, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    if img.shape[0] != 128 or img.shape[1] != 256:
        img = tf.image.resize(img, (128, 256))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# %%
def prepare_dataset(independent, dependent):
    # WCZYTANIE PLIKU CSV INTEGRUJACEGO DANE
    if(VERBOSE == 1):
        print(f"\nRozpoczynam przygotowanie zbioru danych {independent} -> {dependent}...")
    df = pd.read_csv("../dataset/dataset_index_encoded_equalized.csv")
    if(N_ROWS != -1):
        df = df.iloc[:N_ROWS,:]
    if(INCLUDE_INVALID == False):
        df = df[~df[independent].str.contains("invalid", na=False)]

    # PRZEJSCIE NA TYP numpy.ndarray
    data = df[[independent, dependent]].to_numpy()
    if(VERBOSE == 1):
        print("Informacja o zrownowazeniu szykowanego zbioru danych: ")
        print(np.bincount(data[:,1].astype(int)))

    # PODZIAL NA ZBIOR TRENINGOWY I TESTOWY
    X = data[:,0]
    y = data[:,1].astype(np.int32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.60,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    if(VERBOSE == 1):
        print(f"Rozmiar tensora danych treningowych  : {X_train.shape}")
        print(f"Rozmiar tensora danych walidacyjnych : {X_val.shape}")
        print(f"Rozmiar tensora danych testowych     : {X_test.shape}")

    # PRZEJSCIE NA TENSORFLOW
    if independent.contains("spec"):
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(seed=RANDOM_STATE)
            .map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_dataset = (
            tf.data.Dataset.from_tensor_slices((X_test, y_test))
            .map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    else:
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(10000)
            .map(load_numpy, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .map(load_numpy, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_dataset = (
            tf.data.Dataset.from_tensor_slices((X_test, y_test))
            .map(load_numpy, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        # STANDARYZACJA TYLKO DLA DANYCH Z PLIKOW .numpy
        normalizer = Normalization(axis=-1)
        normalizer.adapt(train_dataset.map(lambda x, y: x))
        train_dataset = train_dataset.map(lambda x, y: (normalizer(x), y))
        test_dataset = test_dataset.map(lambda x, y: (normalizer(x), y))
        val_dataset = val_dataset.map(lambda x, y: (normalizer(x), y))

    if(VERBOSE == 1):
        print("Rozmiary tensorow zmiennych niezaleznych (X) i zaleznych (y):")
        x, y = next(iter(train_dataset))
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        print(f"Zakonczylem przygotowywanie zbioru danych {independent} -> {dependent}.\n")
    return train_dataset, val_dataset, test_dataset

# %%
### TRENING I ZAPIS WYNIKOW
n_models = len(MODELS)

for i,config in enumerate(MODELS,start=1):
    # INFORMACJE O MODELU
    print(f"\n\n##### --- TRENING MODELU {i}/{n_models} --- #####")
    name = config["name"]
    independent_variable = config["independent_variable"]
    dependent_variable = config["dependent_variable"]
    loader = config["loader"]
    print("\tInformacje o trenowanym modelu:")
    print(f"\tNazwa: {name}")
    print(f"\tZmienna niezalezna: {independent_variable}")
    print(f"\tZmienna zalezna: {dependent_variable}")

    # PRZYGOTOWANIE ZBIORU DANYCH
    train_dataset, val_dataset, test_dataset = prepare_dataset(independent_variable, dependent_variable)

    # ZALADOWANIE MODELU
    model = getattr(models, loader)()
    
    # ZALADOWANIE WYWOLAN ZWROTNYCH
    callbacks = models.load_callbacks(name)

    # ZAPIS PODSUMOWANIA MODELU DO PLIKU TEKSTOWEGO
    with open("../results/summaries/"+name+"_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # WYPISANIE PODSUMOWANIA NA EKRAN
    if(VERBOSE == 1):
        model.summary()

    # TRENING MODELU
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n\nZakonczono proces treningu modelu {i}/{n_models}. Najlepsze wagi zostaly zapisane w pliku.")

    # ZAPIS HISTORII TRENINGU DO PLIKU
    pd.DataFrame(history.history).to_csv(
        "../results/training/"+name+"_training_history.csv",
        index=False
    )

    # WCZYTANIE NAJLEPSZEGO MODELU I PREDYKCJA
    best_model = load_model("weights/best_"+name+"_model.keras")

    y_test = np.concatenate([
        y.numpy()
        for _, y in test_dataset
    ])
    y_proba = best_model.predict(test_dataset)

    # FRAGMENT KODU DLA KLASYFIKACJI BINARNEJ
    if dependent_variable == "gender":
        y_pred = (y_proba.squeeze() >= 0.5).astype(np.int32)
        cm = confusion_matrix(y_test, y_pred)

        # WYZNACZENIE METRYK
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        roc_auc = roc_auc_score(y_test, y_proba.squeeze())

        # ZAPIS DO PLIKU TEKSTOWEGO
        lines = []
        lines.append("Accuracy    :", accuracy_score(y_test, y_pred))
        lines.append("Precision   :", precision_score(y_test, y_pred))
        lines.append("Sensivity   :", recall_score(y_test, y_pred))
        lines.append("Specificity :",specificity)
        lines.append("F1-score    :", f1_score(y_test, y_pred))
        lines.append("ROC-AUC     :", roc_auc)
        with open("../results/metrics/"+name+".txt", "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")

    # FRAGMENT KODU DLA KLASYFIKACJI WIELOKLASOWEJ
    else:
        y_pred = np.argmax(y_proba, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        content = classification_report(y_test, y_pred)
        with open("../results/metrics/"+name+".txt", "w", encoding="utf-8") as f:
            f.write(content + "\n")
        
    
    print("Metryki zapisane do pliku.\n")
    np.save("../results/confusion_matrixes/"+name+"_cm.npy", cm)
    print("Macierz pomylek zapisana do pliku.\n")

    if(VERBOSE == 1):
        print("\nMetryki modelu: ")
        for line in lines:
            print(line)