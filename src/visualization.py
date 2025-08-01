import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(signal, title='Time Series', ax=None):
    """
    Plot raw or processed time-series signal.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(signal, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Amplitude")
    ax.grid(True)


def plot_fft(fft_result, title='FFT Spectrum', ax=None):
    """
    Plot frequency-domain signal.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fft_result['frequency'], fft_result['amplitude'], color='darkred', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)


def plot_spectrogram(spec_result, title='Spectrogram', ax=None):
    """
    Plot time-frequency spectrogram.
    """
    if ax is None:
        fig, ax = plt.subplots()
    f = spec_result['frequencies']
    t = spec_result['times']
    Sxx = spec_result['Sxx']
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(pcm, ax=ax, label="Power (dB)")


def plot_anomalies(signal, anomaly_mask, title='Anomaly Detection', ax=None):
    """
    Plot time series with anomalies highlighted.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(signal, label='Signal', linewidth=1)
    ax.scatter(np.where(anomaly_mask)[0], signal[anomaly_mask], color='red', s=10, label='Anomalies')
    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
