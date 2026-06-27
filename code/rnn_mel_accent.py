# %%
# NIEZBEDNE IMPORTY

from keras import layers as layer
from keras.models import Sequential
from keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint

# *** USTAWIENIA ***
#n_timesteps = 336 # -> Liczba kroków czasu
#n_features = 256 # -> Liczba cech
#n_outputs = 1 # -> Liczba klas

# %%
def load_model():
    rnn_accent_mel_model = Sequential([
        layer.Bidirectional(
            layer.LSTM(256, return_sequences=True), input_shape=(336, 256)
        ),
        layer.Dropout(0.3),
        layer.Bidirectional(
            layer.LSTM(128)
        ),
        layer.Dropout(0.3),
        layer.Dense(64, activation='relu'),
        layer.Dense(16, activation='softmax')
    ])

    rnn_accent_mel_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            AUC(
                name="roc_auc",
                curve="ROC",
                from_logits=False
            )
        ]
    )

    return rnn_accent_mel_model

def load_callbacks():
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath='weights/best_rnn_mel_accent_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    return [early_stopping, checkpoint]