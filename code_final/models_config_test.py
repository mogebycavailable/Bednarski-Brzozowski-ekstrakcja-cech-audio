MODELS = [
    {
        "name": "mlp_gender_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "gender",
        "loader": "load_model_MLPGenderMel"
    },

    {
        "name": "rnn_gender_mfcc",
        "independent_variable": "mfcc_eq_path",
        "dependent_variable": "gender",
        "loader": "load_model_RNNGenderMFCC"
    },

    {
        "name": "cnn_gender_mel",
        "independent_variable": "mel_spec_path",
        "dependent_variable": "gender",
        "loader": "load_model_CNNGenderMel"
    },
]