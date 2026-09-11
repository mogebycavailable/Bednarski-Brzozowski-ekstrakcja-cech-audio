# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Pliki CSV oraz odpowiadające im nazwy modeli
files = {
    "MLP/Mel": "../results_and_weights/results/training/mlp_accent_mel_training_history.csv",
    "MLP/MFCC": "../results_and_weights/results/training/mlp_accent_mfcc_training_history.csv",
    "CNN/Mel": "../results_and_weights/results/training/cnn_accent_mel_training_history.csv",
    "CNN/MFCC": "../results_and_weights/results/training/cnn_accent_mfcc_training_history.csv",
    "BiLSTM/Mel": "../results_and_weights/results/training/rnn_accent_mel_training_history.csv",
    "BiLSTM/MFCC": "../results_and_weights/results/training/rnn_accent_mfcc_training_history.csv",
}

# %%
# =========================
# VAL ACCURACY
# =========================

plt.figure(figsize=(8, 6))

for model, file in files.items():
    data = pd.read_csv(file)

    plt.plot(
        data.index + 1,
        data["val_accuracy"],
        #marker="o",
        label=model
    )

plt.xlabel("Epoka")
plt.ylabel("Dokładność walidacyjna modeli")
#plt.title("Dokładność walidacyjna modeli")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# =========================
# VAL LOSS
# =========================

plt.figure(figsize=(8, 6))

for model, file in files.items():
    data = pd.read_csv(file)

    plt.plot(
        data.index + 1,
        data["val_loss"],
        #marker="o",
        label=model
    )

plt.xlabel("Epoka")
plt.ylabel("Wartość funkcji straty dla zbioru walidacyjnego")
#plt.title("Wartość funkcji straty dla zbioru walidacyjnego")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
