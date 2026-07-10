# %%
# NIEZBEDNE IMPORTY

from keras import layers as layer
from keras.models import Sequential
from keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint

# *** USTAWIENIA ***
#n_timesteps = 336 # -> Liczba kroków czasu
#n_features = 20 # -> Liczba cech
#n_outputs = 16 # -> Liczba klas

# %%
def load_model():
    rnn_accent_mfcc_model = Sequential([
        layer.LayerNormalization(input_shape=(336, 20)),

        layer.Bidirectional(
            layer.LSTM(
                128,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            )
        ),

        layer.LayerNormalization(),
        layer.Dropout(0.3),

        layer.Bidirectional(
            layer.LSTM(
                64,
                dropout=0.2,
                recurrent_dropout=0.2
            )
        ),
        layer.LayerNormalization(),
        layer.Dropout(0.3),

        layer.Dense(64, activation=None),
        layer.BatchNormalization(),
        layer.Activation("relu"),
        layer.Dropout(0.4),
        layer.Dense(16, activation="softmax")
    ])

    rnn_accent_mfcc_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return rnn_accent_mfcc_model

def load_callbacks():
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath='weights/best_mfcc_rnn_accent_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    return [early_stopping, checkpoint]