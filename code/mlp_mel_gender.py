# %%
# NIEZBEDNE IMPORTY

from keras import layers as layer
from keras.models import Sequential
from keras import regularizers
from keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
# BUDOWA, KOMPILACJA I INFORMACJE O MODELU

def load_model():
    mlp_gender_mel_model = Sequential([
        layer.Input(shape=(336, 256)),
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

        layer.Dense(1, activation='sigmoid')
    ])

    mlp_gender_mel_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(
                name="roc_auc",
                curve="ROC",
                from_logits=False
            )
        ]
    )

    return mlp_gender_mel_model

def load_callbacks():
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath='weights/best_mlp_mel_gender_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    return [early_stopping, checkpoint]
# %%
