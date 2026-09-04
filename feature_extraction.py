import librosa
import numpy as np


def extract_features(file_path):
    """
    Extract MFCC, chroma and mel-spectrogram features
    from an audio file.
    """

    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Chroma features
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate
    )

    chroma_mean = np.mean(chroma.T, axis=0)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate
    )

    mel_mean = np.mean(mel.T, axis=0)

    # Combine all features
    features = np.hstack([
        mfcc_mean,
        chroma_mean,
        mel_mean
    ])

    return features