# %%
# NIEZBEDNE IMPORTY

from keras import layers as layer
from keras.models import Sequential
from keras import regularizers
from keras.metrics import Precision
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
# BUDOWA, KOMPILACJA I INFORMACJE O MODELU

def load_model():
    mlp_accent_mel_model = Sequential([
        layer.Input(shape=(256, 336)),
        layer.Flatten(),

        layer.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layer.BatchNormalization(),
        layer.Dropout(0.4),

        layer.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layer.BatchNormalization(),
        layer.Dropout(0.4),

        layer.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layer.Dropout(0.3),

        layer.Dense(64, activation='relu'),
        layer.Dropout(0.2),

        layer.Dense(16, activation='softmax')
    ])

    mlp_accent_mel_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return mlp_accent_mel_model

def load_callbacks():
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath='weights/best_mel_mlp_accent_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    return [early_stopping, checkpoint]


# %%
