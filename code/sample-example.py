# %%
import numpy as np
import librosa
from audio_booster import booster
from preprocessing_checklist import *
import joblib

from keras.models import load_model

from sklearn.preprocessing import StandardScaler

# %%
# check-sample.mp3
# danuta.mp3
# andrzej.mp3
x, sr = librosa.load('../common-voice/cv-invalid/cv-invalid/sample-000002.mp3', sr=None)
print("Czestotliwosc probkowania: "+str(sr))
x = booster(x)
mel_num = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=256)
#print(mel_num)
print(mel_num.shape)

# %%
with open("scalers/mel_num_scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

x_eq = equalize_shape_in_hand(mel_num, 336)
print("Rozmiar po equalizacji: "+str(x_eq.shape))

x_flat = x_eq.reshape(1,-1)
print(x_flat.shape)
x_flat_scaled = scaler.transform(x_flat)
x_ready = x_flat_scaled.reshape(1,256,336)

# %%
model = load_model("weights/best_mel_mlp_weights.keras")

y_proba = model.predict(x_ready)

print("P-stwo, ze to mezczyzna: "+str(y_proba))

y_pred = (y_proba > 0.5).astype(int)

print("Predicted class:", y_pred[0][0])
label_map = {0: "female", 1: "male"}

print("Predicted label:", label_map[y_pred[0][0]])


# %%
