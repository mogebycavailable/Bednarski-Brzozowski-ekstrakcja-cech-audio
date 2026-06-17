# %%
# NIEZBEDNE IMPORTY

from keras import layers as layer
from keras.models import Sequential
from keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
def load_model():
    cnn_gender_mfcc_model = Sequential([
        layer.Input(shape=(128, 256, 1)),

        layer.Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"),
        layer.BatchNormalization(),
        layer.MaxPooling2D(pool_size=(2,2)),

        layer.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        layer.BatchNormalization(),
        layer.MaxPooling2D(pool_size=(2,2)),

        layer.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        layer.BatchNormalization(),
        layer.MaxPooling2D(pool_size=(2,2)),

        layer.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        layer.BatchNormalization(),

        layer.GlobalAveragePooling2D(),

        layer.Dense(64, activation='relu'),
        layer.Dropout(0.5),

        layer.Dense(1, activation='sigmoid')
    ])

    cnn_gender_mfcc_model.compile(
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

    return cnn_gender_mfcc_model

def load_callbacks():
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath='weights/best_mfcc_cnn_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    return [early_stopping, checkpoint]
# %%
