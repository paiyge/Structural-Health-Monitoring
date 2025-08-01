# Structural Health Monitoring Pipeline

This project implements a modular, reproducible Python-based pipeline for structural health monitoring using time-series sensor data. The system simulates or loads vibration/thermal/shock data, applies signal processing and anomaly detection techniques, and generates visualizations and reports.

---

## Features

- Modular architecture (importable Python modules)
- Time-series preprocessing (normalization, resampling)
- Signal processing (FFT, bandpass filters, spectrograms)
- Anomaly detection (Isolation Forest, PCA, z-score)
- Scientific plots and analysis reports
- Jupyter notebook for interactive experimentation

---

## Project Structure
#### Notebooks 
- StructuralHealthMonitoring_Demo.ipynb
#### Python modules
-  data_loader.py
-  preprocessing.py
-  signal_processing.py
-  anomaly_detection.py
-  visualization.py
-  main.py

#### Other
- requirements.txt # Python dependencies
- README.md

## Outputs saved with a timestamp in the reports/ folder
- Time-domain signal plot
- FFT plot
- Spectrogram
- Anomaly detection overlay
