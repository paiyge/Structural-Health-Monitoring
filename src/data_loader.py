# structural_health_pipeline/main.py

from src.data_loader import load_nasa_bearing_data
from src.preprocessing import preprocess_data
from src.signal_processing import apply_fft, apply_bandpass_filter, compute_spectrogram
from src.anomaly_detection import detect_anomalies_isolation_forest
from src.visualization import plot_time_series, plot_fft, plot_spectrogram, plot_anomalies
import matplotlib.pyplot as plt
import os


def main():
    # Step 1: Load data
    data, sampling_rate = load_nasa_bearing_data("data/bearing_data")

    # Step 2: Preprocess
    cleaned_data = preprocess_data(data)

    # Step 3: Signal processing
    filtered_data = apply_bandpass_filter(cleaned_data, lowcut=500, highcut=2000, fs=sampling_rate)
    fft_data = apply_fft(filtered_data, fs=sampling_rate)
    spectrograms = compute_spectrogram(filtered_data, fs=sampling_rate)

    # Step 4: Anomaly detection
    anomalies = detect_anomalies_isolation_forest(filtered_data)

    # Step 5: Visualizations
    os.makedirs("reports", exist_ok=True)
    for sensor_id, signal in filtered_data.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_time_series(signal, title=f"Sensor {sensor_id} - Time Series", ax=ax)
        plt.savefig(f"reports/sensor_{sensor_id}_timeseries.png")

        fig, ax = plt.subplots()
        plot_fft(fft_data[sensor_id], title=f"Sensor {sensor_id} - FFT", ax=ax)
        plt.savefig(f"reports/sensor_{sensor_id}_fft.png")

        fig, ax = plt.subplots()
        plot_spectrogram(spectrograms[sensor_id], title=f"Sensor {sensor_id} - Spectrogram", ax=ax)
        plt.savefig(f"reports/sensor_{sensor_id}_spectrogram.png")

        fig, ax = plt.subplots()
        plot_anomalies(signal, anomalies[sensor_id], title=f"Sensor {sensor_id} - Anomalies", ax=ax)
        plt.savefig(f"reports/sensor_{sensor_id}_anomalies.png")


if __name__ == "__main__":
    main()
