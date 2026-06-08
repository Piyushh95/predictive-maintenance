# 🏭 Predictive Maintenance Control Center
<img width="1920" height="1080" alt="Screenshot 2026-05-14 002328" src="https://github.com/user-attachments/assets/c0492505-e15c-4f87-bfa3-cd781de12c0b" />


A real-time machine health monitoring system that analyzes multivariate sensor data to detect anomalies, predict equipment failures, and estimate Remaining Useful Life (RUL) — all surfaced through a live auto-refreshing Streamlit dashboard.

---

## 🖥️ Dashboard Preview<img width="1920" height="1080" alt="Screenshot 2026-05-14 002403" src="https://github.com/user-attachments/assets/302edcd1-266e-4f15-9c6f-f24cbcad5e95" />
<img width="1920" height="1080" alt="Screenshot 2026-05-14 002352" src="https://github.com/user-attachments/assets/e30992ba-d066-44ae-b14e-732ee48c68b8" />


> Live dashboard running at `localhost:8502`

**KPI Cards (real-time):**

| Metric | Example Value |
|---|---|
| Latest Health Score | 12.6 / 100 |
| Machine Status | 🔴 EMERGENCY |
| Failure Probability | 42.0% |
| RUL (hours) | 0.0 h |

**Dashboard Tabs:** `Live Monitoring` · `Failure Prediction` · `Root Cause Insights`

**Sub-tabs:** `Overview` · `Diagnostics` · `Reports`

---

## 🚀 Problem

Unplanned machine failures cause costly downtime. This system detects degradation early — giving engineers time to act before failure occurs.

**Pipeline:**
```
Sensor Data → Feature Engineering → Anomaly Detection → Health Scoring → RUL Estimation → Alert + Report
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| ML | `Scikit-learn` (Isolation Forest, Random Forest) |
| Data | `Pandas`, `NumPy` |
| Feature Engineering | Rolling stats, degradation slopes |
| Dashboard | `Streamlit` (auto-refresh: 15 sec) |
| Visualization | `Plotly`, `Matplotlib` |
| Reporting | Text report generation + download |

---

## 🔑 Key Features

- **Real-time health monitoring** — dashboard auto-refreshes every 15 seconds
- **Health Score (0–100)** — continuous degradation score calculated from sensor readings
- **Alert classification** — Normal → Warning → Critical → Emergency with live badge display
- **Failure probability prediction** — ML model outputs probability of imminent failure
- **RUL estimation** — estimates how many hours until failure threshold is reached
- **Degradation diagnostics** — rolling slope analysis to detect when degradation begins (e.g. degradation start index: 27)
- **Root Cause Analysis** — feature importance chart showing which sensors (pressure, temperature, vibration) drive failure risk most
- **Alert level filtering** — filter dashboard view by alert severity
- **Time range slider** — explore any historical window of sensor data
- **Failure report generation** — one-click detailed machine risk report download as `.txt`

---

## 📊 Example Output

```
Health Score:        12.6  (EMERGENCY)
Machine Status:      EMERGENCY
Failure Probability: 42.0%
Estimated RUL:       0.0 hours
Degradation Start:   Index 27
Predicted Failure:   2024-01-01 16:39:00
Top Risk Factors:    pressure_mean (0.31), temperature_mean (0.20), temperature (0.18)
```

---

## 🗂️ Project Structure

```
predictive-maintenance/
├── main.py               # Core prediction workflow
├── model.py              # Model training and inference
├── features.py           # Feature engineering logic
├── utils.py              # Helper functions
├── dashboard.py          # Streamlit dashboard (multi-tab, auto-refresh)
├── generate_data.py      # Simulated sensor data generation
├── data/
│   └── sensor_data.csv   # Input sensor dataset
├── machine_report.txt    # Generated machine risk report
└── requirements.txt
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Dashboard opens at `localhost:8502` and auto-refreshes every 15 seconds.

---

## 💡 What I'd Improve Next

- Replace simulated data with real industrial dataset (NASA CMAPSS / PHM08)
- Add automated retraining pipeline when new sensor data arrives
- Deploy as a REST API (FastAPI) for integration with real SCADA/PLC systems
- Add email/SMS alert triggers on Emergency status
- Multi-machine monitoring (fleet view)

---

## ⚠️ Disclaimer

Built for educational and prototyping purposes. Predictions should support — not replace — engineering judgment in production environments.
