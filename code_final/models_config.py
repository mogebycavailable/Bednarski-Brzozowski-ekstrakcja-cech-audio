MODELS = [
    
    # MLP MEL
    {
        "name": "mlp_gender_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "gender",
        "loader": "load_model_MLPGenderMel"
    },
    {
        "name": "mlp_age_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "age",
        "loader": "load_model_MLPAgeMel"
    },
    
    {
        "name": "mlp_accent_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "accent",
        "loader": "load_model_MLPAccentMel"
    },
    
    # MLP MFCC
    {
        "name": "mlp_gender_mfcc",
        "independent_variable": "mfcc_eq_path",
        "dependent_variable": "gender",
        "loader": "load_model_MLPGenderMFCC"
    },
    {
        "name": "mlp_age_mfcc",
        "independent_variable": "mfcc_eq_path",
        "dependent_variable": "age",
        "loader": "load_model_MLPAgeMFCC"
    },

    {
        "name": "mlp_accent_mfcc",
       "independent_variable": "mfcc_eq_path",
        "dependent_variable": "accent",
        "loader": "load_model_MLPAccentMFCC"
    },

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

    # CNN MEL
    {
        "name": "cnn_gender_mel",
        "independent_variable": "mel_spec_path",
        "dependent_variable": "gender",
        "loader": "load_model_CNNGenderMel"
    },

    {
        "name": "cnn_age_mel",
        "independent_variable": "mel_spec_path",
        "dependent_variable": "age",
        "loader": "load_model_CNNAgeMel"
    },
    {
        "name": "cnn_accent_mel",
        "independent_variable": "mel_spec_path",
        "dependent_variable": "accent",
        "loader": "load_model_CNNAccentMel"
    },

    # CNN MFCC
    {
        "name": "cnn_gender_mfcc",
        "independent_variable": "mfcc_spec_path",
        "dependent_variable": "gender",
        "loader": "load_model_CNNGenderMFCC"
    },
    {
        "name": "cnn_age_mfcc",
        "independent_variable": "mfcc_spec_path",
        "dependent_variable": "age",
        "loader": "load_model_CNNAgeMFCC"
    },
    {
        "name": "cnn_accent_mfcc",
        "independent_variable": "mfcc_spec_path",
        "dependent_variable": "accent",
        "loader": "load_model_CNNAccentMFCC"
    }

]