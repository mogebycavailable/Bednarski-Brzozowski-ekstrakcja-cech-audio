# %%
import numpy as np
import librosa
from audio_booster import booster
from preprocessing_checklist import *
import joblib
import PIL as image

from keras.models import load_model

from sklearn.preprocessing import StandardScaler

# %%
# check-sample.mp3
# danuta.mp3
# andrzej.mp3
x, sr = librosa.load('../danuta.mp3', sr=None)
print("Czestotliwosc probkowania: "+str(sr))
x = booster(x)
mel_num = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=256)
#print(mel_num)
print(mel_num.shape)

# %%

mel_db = librosa.power_to_db(mel_num, ref=np.max)
fig = plt.figure(figsize=(512/128,256/128), dpi=128)
ax = fig.add_axes([0, 0, 1, 1])
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='m', cmap="gray")
ax.set_axis_off()
plt.savefig("../sample_example.png", pad_inches=0, dpi=128)
plt.show()
plt.close()

# %%
'''
with open("scalers/mel_num_scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

x_eq = equalize_shape_in_hand(mel_num, 336)
print("Rozmiar po equalizacji: "+str(x_eq.shape))

x_flat = x_eq.reshape(1,-1)
print(x_flat.shape)
x_flat_scaled = scaler.transform(x_flat)
x_ready = x_flat_scaled.reshape(1,256,336)
'''

# %%
model = load_model("weights/best_mel_cnn_model.keras")

def load(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.resize(img, (128,256))
    img = tf.cast(img, tf.float16) / 255.0
    return img

spec = load("../sample_example.png")
spec = np.expand_dims(spec, axis=0)

#spec = tf.expand_dims(spec, axis=-1)
spec.shape
# %%

y_proba = model.predict(spec)

print("P-stwo, ze to mezczyzna: "+str(y_proba))

y_pred = (y_proba > 0.5).astype(int)

print("Predicted class:", y_pred[0][0])
label_map = {0: "female", 1: "male"}

print("Predicted label:", label_map[y_pred[0][0]])


# %%
