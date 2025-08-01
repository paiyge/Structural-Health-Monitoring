import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt


def apply_fft(signals, fs=1000):
    """
    Compute FFT of each signal.
    Args:
        signals (dict): Dictionary of signals (e.g., 'vibration': np.array)
        fs (int): Sampling frequency in Hz
    Returns:
        dict: Dictionary of FFT results with frequency and amplitude
    """
    fft_results = {}
    for key, signal in signals.items():
        n = len(signal)
        freqs = np.fft.fftfreq(n, d=1/fs)
        fft_vals = np.abs(fft(signal))[:n // 2]
        fft_freqs = freqs[:n // 2]
        fft_results[key] = {'frequency': fft_freqs, 'amplitude': fft_vals}
    return fft_results


def apply_bandpass_filter(signals, lowcut, highcut, fs=1000, order=4):
    """
    Apply Butterworth bandpass filter to each signal.
    Args:
        signals (dict): Dictionary of raw signals
        lowcut (float): Lower cutoff frequency (Hz)
        highcut (float): Upper cutoff frequency (Hz)
        fs (int): Sampling frequency
        order (int): Filter order
    Returns:
        dict: Filtered signals
    """
    filtered = {}
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    for key, signal in signals.items():
        filtered[key] = filtfilt(b, a, signal)
    return filtered


def compute_spectrogram(signals, fs=1000, nperseg=256, noverlap=128):
    """
    Compute spectrogram for each signal.
    Args:
        signals (dict): Dictionary of signals
        fs (int): Sampling frequency
    Returns:
        dict: Dictionary with 'frequencies', 'times', and 'spectrogram'
    """
    specs = {}
    for key, signal in signals.items():
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        specs[key] = {'frequencies': f, 'times': t, 'Sxx': Sxx}
    return specs


# Placeholder for optional wavelet analysis
def apply_wavelet_transform(signal):
    """
    Optional: Add wavelet transform (e.g., using pywt) if desired.
    """
    pass  # Future implementation
