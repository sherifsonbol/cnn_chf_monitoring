import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import datetime

# === Load model ===
hr_model = load_model("models/cnn_lstm_hr_model.keras")

# === Page title ===
st.title("PPG Monitoring Dashboard")

# === Patient Info ===
st.sidebar.header("Patient Information")
patient_name = st.sidebar.text_input("Patient Name", "")
patient_id = st.sidebar.text_input("Patient ID", "")

# === Upload file ===
uploaded_file = st.file_uploader("Upload PPG live recording (.npy)", type=["npy"])

if uploaded_file is not None:
    X_live = np.load(uploaded_file)

    if X_live.ndim != 2 or X_live.shape[1] != 512:
        st.error(f"Uploaded file shape {X_live.shape} is invalid. Expected (n_windows, 512).")
    else:
        # Parameters
        hr_threshold_percent = 0.10  # 10% change
        hrv_threshold_percent = 0.20 # 20% change
        consecutive_alert_windows = 2
        smoothing_windows = 3  # Average over 3 windows

        baseline_hr = None
        baseline_hrv = None
        bad_window_count = 0

        hr_list = []
        hrv_list = []
        status_list = []

        results = []

        st.subheader("Monitoring Results")

        for i in range(len(X_live)):
            window = X_live[i]

            # Predict
            pred_hr = hr_model.predict(window.reshape(1, 512, 1))[0][0]
            pred_hrv = np.std(window)

            hr_list.append(pred_hr)
            hrv_list.append(pred_hrv)

            # Smoothing over last 3 windows
            avg_hr = np.mean(hr_list[-smoothing_windows:])
            avg_hrv = np.mean(hrv_list[-smoothing_windows:])

            if baseline_hr is None:
                baseline_hr = avg_hr
                baseline_hrv = avg_hrv

            delta_hr = abs(avg_hr - baseline_hr) / baseline_hr
            delta_hrv = abs(avg_hrv - baseline_hrv) / baseline_hrv

            is_hr_alert = delta_hr > hr_threshold_percent
            is_hrv_alert = delta_hrv > hrv_threshold_percent

            if is_hr_alert or is_hrv_alert:
                bad_window_count += 1
            else:
                bad_window_count = 0

            # Decide alert
            if bad_window_count >= consecutive_alert_windows:
                alert = "RED"
            elif is_hr_alert or is_hrv_alert:
                alert = "YELLOW"
            else:
                alert = "STABLE"

            status_list.append(alert)

            results.append({
                "Window": i + 1,
                "HR (bpm)": f"{avg_hr:.1f}",
                "HRV": f"{avg_hrv:.1f}",
                "Status": alert
            })

        # === Display Table ===
        df_results = pd.DataFrame(results)

        def color_status(val):
            color = "green" if val == "STABLE" else "yellow" if val == "YELLOW" else "red"
            return f"background-color: {color}; color: black;"

        st.dataframe(
            df_results.style.applymap(color_status, subset=["Status"]),
            use_container_width=True
        )

        # === Download Button ===
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"ppg_monitoring_{now}.csv"

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Monitoring Results",
            data=csv,
            file_name=csv_filename,
            mime="text/csv"
        )

        # === Trend Charts ===
        st.subheader("Trends Over Time")

        fig_hr, ax_hr = plt.subplots()
        colors = ["green" if s == "STABLE" else "yellow" if s == "YELLOW" else "red" for s in status_list]
        ax_hr.scatter(range(1, len(hr_list) + 1), hr_list, c=colors)
        ax_hr.set_title("Heart Rate Trend")
        ax_hr.set_xlabel("Window Number")
        ax_hr.set_ylabel("Heart Rate (bpm)")
        st.pyplot(fig_hr)

        fig_hrv, ax_hrv = plt.subplots()
        ax_hrv.scatter(range(1, len(hrv_list) + 1), hrv_list, c=colors)
        ax_hrv.set_title("HRV Trend")
        ax_hrv.set_xlabel("Window Number")
        ax_hrv.set_ylabel("HRV (std dev)")
        st.pyplot(fig_hrv)

        st.success("✅ Monitoring complete!")

else:
    st.info("Please upload a live PPG file (.npy)")
