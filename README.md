# PPG Monitoring and CHF Risk Dashboard

This project provides a real-time monitoring dashboard for patients with Congestive Heart Failure (CHF) using PPG (Photoplethysmography) signals.

## Features

- 📈 Predicts Heart Rate (HR) and Heart Rate Variability (HRV) from PPG signals
- 🚦 Adaptive alert system: STABLE / YELLOW / RED
- 🧠 Uses a CNN+LSTM model for robust HR prediction
- 📊 Displays trend charts over time
- 💾 Downloadable monitoring reports (.csv)
- 🏥 Designed for post-discharge monitoring of CHF patients

## How It Works

- Upload a `.npy` file containing PPG signal windows.
- The app predicts HR and HRV for each window.
- Alerts are triggered based on deviation from baseline.
- Data and alerts are displayed in an interactive dashboard.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
Install the required packages:
pip install -r requirements.txt
Run the App
streamlit run monitor_ppg_streamlit.py
Make sure the model file models/cnn_lstm_hr_model.keras exists.
Folder Structure
├── models/
│   └── cnn_lstm_hr_model.keras
├── monitor_ppg_streamlit.py
├── requirements.txt
├── README.md

