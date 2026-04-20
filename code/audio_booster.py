import numpy as np

def booster(audio):
    max_amp = np.max(np.abs(audio))
    if(max_amp==0):
        return audio
    gain = 1.0 / max_amp
    if(gain > 1.0):
        return audio * gain
    else:
        return audio