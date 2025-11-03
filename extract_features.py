import pandas as pd
import numpy as np
import librosa

def extract_features(filename):
    y, sr = librosa.load(filename, sr=None, mono=True, duration=120)
    if y.shape[0] < 22050:
        return None

    features = {}
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)
    features['tempo_strength'] = float(librosa.beat.plp(y=y, sr=sr).mean())
    y_harm, y_perc = librosa.effects.hpss(y)
    features['harmonic_mean'] = float(np.mean(np.abs(y_harm)))
    features['percussive_mean'] = float(np.mean(np.abs(y_perc)))
    features['harm_perc_ratio'] = float(np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-6))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = chroma.mean(axis=1).tolist()
    features['chroma_std'] = chroma.std(axis=1).tolist()
    chroma_avg = chroma.mean(axis=1)
    key_index = chroma_avg.argmax()
    major_template = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_template = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    major_score = np.corrcoef(chroma_avg, np.roll(major_template, key_index))[0, 1]
    minor_score = np.corrcoef(chroma_avg, np.roll(minor_template, key_index))[0, 1]
    mode = 0 if major_score > minor_score else 1
    features['key_mode'] = int(key_index + 12 * mode)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = mfcc.mean(axis=1).tolist()
    features['mfcc_std'] = mfcc.std(axis=1).tolist()
    delta_mfcc = librosa.feature.delta(mfcc)
    features['delta_mfcc_mean'] = delta_mfcc.mean(axis=1).tolist()
    features['delta_mfcc_std'] = delta_mfcc.std(axis=1).tolist()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spec_centroid_mean'] = float(centroid.mean())
    features['spec_bandwidth_mean'] = float(bandwidth.mean())
    features['spec_rolloff_mean'] = float(rolloff.mean())
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['contrast_mean'] = contrast.mean(axis=1).tolist()
    features['contrast_std'] = contrast.std(axis=1).tolist()
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features['tonnetz_mean'] = tonnetz.mean(axis=1).tolist()
    features['tonnetz_std'] = tonnetz.std(axis=1).tolist()
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(zcr.mean())
    features['zcr_std'] = float(zcr.std())
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = float(rms.mean())
    features['rms_std'] = float(rms.std())
    features['duration'] = float(librosa.get_duration(y=y, sr=sr))

    # Flatten lists to separate values
    flat_features = {}
    for k, v in features.items():
        if isinstance(v, list):
            for i, val in enumerate(v):
                flat_features[f"{k}_{i}"] = val
        else:
            flat_features[k] = v

    return pd.DataFrame([flat_features])

