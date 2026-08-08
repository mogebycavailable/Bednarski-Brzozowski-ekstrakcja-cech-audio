MODELS = [

    {
        "name": "rnn_accent_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "accent",
        "loader": "load_model_RNNAccentMel"
    },

    {
        "name": "cnn_accent_mel",
        "independent_variable": "mel_spec_path",
        "dependent_variable": "accent",
        "loader": "load_model_CNNAccentMel"
    },

        {
        "name": "mlp_accent_mel",
        "independent_variable": "mel_eq_path",
        "dependent_variable": "accent",
        "loader": "load_model_MLPAccentMel"
    },

]