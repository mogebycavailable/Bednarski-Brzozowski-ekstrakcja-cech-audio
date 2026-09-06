import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from class_labels import LABELS_GENDER, LABELS_ACCENT, LABELS_AGE

### ETYKIETY KLAS
LABELS = {
    "gender": LABELS_GENDER,
    "age": LABELS_AGE,
    "accent": LABELS_ACCENT,
}

### SCIEZKI FOLDEROW
input_folder = Path("../results_and_weights/results/confusion_matrixes")

output_folder = Path('../results_and_weights/results/cm_plots')
output_folder.mkdir(parents=True, exist_ok=True)

normalized_output_folder = Path('../results_and_weights/results/cm_norm_plots')
normalized_output_folder.mkdir(parents=True, exist_ok=True)

### GENEROWANIE MACIERZY POMYLEK Z PLIKOW NUMPY
for file in sorted(input_folder.glob("*.npy")):

    print(f"Przetwarzanie: {file.name}")
    cm = np.load(file)
    filename = file.stem.lower()

    if "gender" in filename:
        labels = LABELS["gender"]
        task = "Gender"

    elif "accent" in filename:
        labels = LABELS["accent"]
        task = "Accent"

    elif "age" in filename:
        labels = LABELS["age"]
        task = "Age"

    else:
        print(f"  -> Pomijam: nie rozpoznano typu macierzy")
        continue

    if cm.ndim != 2:
        print(f"  -> Pomijam: macierz nie jest 2D, shape={cm.shape}")
        continue

    if cm.shape[0] != len(labels) or cm.shape[1] != len(labels):
        print(
            f"  -> UWAGA: rozmiar macierzy {cm.shape} "
            f"nie pasuje do liczby etykiet ({len(labels)})"
        )
        continue

    # Automatyczne konfigurowanie proporcji wykresu
    n_classes = len(labels)
        
    figsize = (
        max(8, n_classes * 0.7),
        max(7, n_classes * 0.7)
    )
    
    fontsize = max(5, min(10, 150 // n_classes))
    label_fontsize = max(6, min(10, 150 // n_classes))

    fig, ax = plt.subplots(figsize=figsize)

    # Wyświetlenie macierzy
    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    fig.colorbar(im, ax=ax)

    ax.set_title(f"Model: {file.stem.removesuffix('_cm')}")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    # Zachowanie tej samej nazwy co plik .npy
    output_file = output_folder / f"{file.stem}.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    print(f"  -> Zapisano: {output_file}")

    ### ROW NORMALIZATION
    row_sums = cm.sum(axis=1, keepdims=True)

    # Normalizacja każdego wiersza
    # np. [80, 20] -> [0.8, 0.2]
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        cm_normalized,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    fig.colorbar(im, ax=ax, label="Proportion")

    ax.set_title(f"Model: {file.stem.removesuffix('_cm')} (Row Normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):

            ax.text(
                j,
                i,
                f"{cm_normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center"
            )


    plt.tight_layout()

    normalized_output_file = (
        normalized_output_folder / f"{file.stem}_norm.png"
    )

    plt.savefig(
        normalized_output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    print(f"  -> Zapisano (normalized): {normalized_output_file}")

print("\nGotowe! Macierze pomylek zostaly wygenerowane i zapisane do folderów.")