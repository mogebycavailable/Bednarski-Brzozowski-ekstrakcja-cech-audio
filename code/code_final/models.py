# %%
# NIEZBEDNE IMPORTY
from keras import layers as layer
from keras.models import Sequential
from keras import regularizers
from keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint

### Modele perceptronu wielowarstwowego (MLP) ###
def load_model_MLPGenderMel():
    model = Sequential([
        layer.Input(shape=(336,256)),
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

    model.compile(
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

    return model

def load_model_MLPAgeMel():
    model = Sequential([
        layer.Input(shape=(336,256)),
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

        layer.Dense(8, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return model

def load_model_MLPAccentMel():
    model = Sequential([
        layer.Input(shape=(336,256)),
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

        layer.Dense(16, activation='sofmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return model

def load_model_MLPGenderMFCC():
    model = Sequential([
        layer.Input(shape=(336,20)),
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

    model.compile(
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

    return model

def load_model_MLPAgeMFCC():
    model = Sequential([
        layer.Input(shape=(336,20)),
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

        layer.Dense(8, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return model

def load_model_MLPAccentMFCC():
    model = Sequential([
        layer.Input(shape=(336,20)),
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

        layer.Dense(16, activation='sofmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return model

### Modele sieci rekurencyjncych (RNN) ###
def load_model_RNNGenderMel():
    model = Sequential([
        layer.LayerNormalization(input_shape=(336, 256)),

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
        layer.Dense(1, activation="sigmoid")
    ])

    model.compile(
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

    return model

def load_model_RNNAgeMel():
    model = Sequential([
        layer.LayerNormalization(input_shape=(336, 256)),

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
        layer.Dense(8, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return model

def load_model_RNNAccentMel():
    model = Sequential([
        layer.LayerNormalization(input_shape=(336, 256)),

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

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return model

def load_model_RNNGenderMFCC():
    model = Sequential([
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
        layer.Dense(1, activation="sigmoid")
    ])

    model.compile(
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

    return model

def load_model_RNNAgeMFCC():
    model = Sequential([
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
        layer.Dense(8, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return model

def load_model_RNNAccentMFCC():
    model = Sequential([
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

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    return model

### Modele sieci konwolucyjnych (CNN) ###
def load_model_CNNGenderMel():
    model = Sequential([
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

        layer.Dense(1, activation='sigmoid')
    ])

    model.compile(
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
    return model

def load_model_CNNAgeMel():
    model = Sequential([
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

        layer.Dense(8, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )
    return model

def load_model_CNNAccentMel():
    model = Sequential([
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

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )
    return model

def load_model_CNNGenderMFCC():
    model = Sequential([
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

        layer.Dense(1, activation='sigmoid')
    ])

    model.compile(
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
    return model

def load_model_CNNAgeMFCC():
    model = Sequential([
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

        layer.Dense(8, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )
    return model

def load_model_CNNAccentMFCC():
    model = Sequential([
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

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )
    return model

### Definicja wywolan zwrotnych ###
def load_callbacks(model_name : str):
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
    )
    filepath = "weights/best_"+model_name+"_model.keras"
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    return [early_stopping, checkpoint]