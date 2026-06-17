import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def train_val_test_split(
    ds,
    target_index=1,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    stratify=True,
    shuffle=True,
    random_state=42
):
    """
    Split tf.data.Dataset into train/val/test datasets.

    Parameters
    ----------
    ds : tf.data.Dataset
        Dataset yielding tuples, e.g. (X, y)

    target_index : int
        Position of target variable in yielded tuple.

    train_size : float
    val_size : float
    test_size : float

    stratify : bool
        Preserve class proportions.

    shuffle : bool

    random_state : int

    Returns
    -------
    train_ds, val_ds, test_ds
    """

    if not np.isclose(
        train_size + val_size + test_size,
        1.0
    ):
        raise ValueError(
            "train_size + val_size + test_size must equal 1"
        )

    samples = list(ds)

    features = []
    labels = []

    for item in samples:
        features.append(item[0])
        labels.append(item[target_index])

    labels = np.array([
        y.numpy() if hasattr(y, "numpy") else y
        for y in labels
    ])

    indices = np.arange(len(labels))

    strat = labels if stratify else None

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        shuffle=shuffle,
        stratify=strat,
        random_state=random_state
    )

    temp_labels = labels[temp_idx]

    val_fraction = val_size / (val_size + test_size)

    strat_temp = temp_labels if stratify else None

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_fraction,
        shuffle=shuffle,
        stratify=strat_temp,
        random_state=random_state
    )

    def build_dataset(idxs):

        x = tf.stack([
            features[i]
            for i in idxs
        ])

        y = tf.convert_to_tensor(
            labels[idxs]
        )

        return tf.data.Dataset.from_tensor_slices(
            (x, y)
        )

    return (
        build_dataset(train_idx),
        build_dataset(val_idx),
        build_dataset(test_idx)
    )