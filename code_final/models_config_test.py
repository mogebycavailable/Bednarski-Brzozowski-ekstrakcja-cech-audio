MODELS = [

    # RNN MEL
    {
        "name": "rnn_gender_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "gender",
        "loader": "load_model_RNNGenderMel"
    },
    {
        "name": "rnn_age_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "age",
        "loader": "load_model_RNNAgeMel"
    },
    {
        "name": "rnn_accent_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "accent",
        "loader": "load_model_RNNAccentMel"
    },

    # RNN MFCC
    {
        "name": "rnn_gender_mfcc",
        "independent_variable": "mfcc_eq_path",
        "dependent_variable": "gender",
        "loader": "load_model_RNNGenderMFCC"
    },
    {
        "name": "rnn_age_mfcc",
        "independent_variable": "mfcc_eq_path",
        "dependent_variable": "age",
        "loader": "load_model_RNNAgeMFCC"
    },
    {
        "name": "rnn_accent_mfcc",
        "independent_variable": "mfcc_eq_path",
        "dependent_variable": "accent",
        "loader": "load_model_RNNAccentMFCC"
    },

]