# %%
# NIEZBEDNE IMPORTY

from keras import layers as layer
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
def load_model():
    cnn_accent_mel_model = Sequential([
        layer.Input(shape=(128, 256, 1)),

        layer.Conv2D(32, kernel_size=(3,3), padding="same", activation=None, use_bias=False),
        layer.BatchNormalization(),
        layer.Activation("relu"),
        layer.MaxPooling2D(pool_size=(2,2)),

        layer.Conv2D(64, kernel_size=(3,3), padding='same', activation=None, use_bias=False),
        layer.BatchNormalization(),
        layer.Activation("relu"),
        layer.MaxPooling2D(pool_size=(2,2)),

        layer.Conv2D(128, kernel_size=(3,3), padding='same', activation=None, use_bias=False),
        layer.BatchNormalization(),
        layer.Activation("relu"),
        layer.MaxPooling2D(pool_size=(2,2)),

        layer.Conv2D(256, kernel_size=(3,3), padding='same', activation=None, use_bias=False),
        layer.BatchNormalization(),
        layer.Activation("relu"),

        layer.GlobalAveragePooling2D(),

        layer.Dense(128, activation='relu'),
        layer.Dropout(0.5),

        layer.Dense(16, activation='softmax')
    ])

    cnn_accent_mel_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return cnn_accent_mel_model

def load_callbacks():
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath='weights/best_mel_cnn_accent_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    return [early_stopping, checkpoint]
# %%
