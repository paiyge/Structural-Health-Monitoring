import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, normalize=True, segment_length=None):
    """
    Preprocess time-series data:
    - Remove NaNs
    - Normalize if specified
    - Segment if segment_length is specified (not required for basic flow)
    
    Args:
        df (DataFrame): Time-series data with columns like 'vibration', 'temperature'.
        normalize (bool): Whether to apply z-score normalization.
        segment_length (int or None): If set, breaks signal into equal-length segments.
    
    Returns:
        dict: A dictionary of preprocessed signals keyed by 'vibration', 'temperature', etc.
    """
    processed = {}

    for col in df.columns:
        if col == 'time':
            continue  # Keep original time vector unmodified

        signal = df[col].dropna().values

        if normalize:
            scaler = StandardScaler()
            signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

        if segment_length:
            signal = segment_signal(signal, segment_length)

        processed[col] = signal

    return processed


def segment_signal(signal, segment_length):
    """
    Break a 1D signal into equal-length segments.
    Pads the end with zeros if necessary.
    """
    total_length = len(signal)
    pad_size = (segment_length - (total_length % segment_length)) % segment_length
    padded = np.pad(signal, (0, pad_size), mode='constant')
    segments = padded.reshape(-1, segment_length)
    return segments
