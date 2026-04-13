# Predictive Maintenance System 🚀

## Problem
Unplanned machine failures cause costly downtime.  
This project predicts failures early using sensor data.

## Solution
An end-to-end predictive maintenance pipeline that includes:
- Feature engineering on multivariate sensor data
- Anomaly detection using Isolation Forest
- Health score calculation (0–100)
- Alert classification (Normal → Emergency)
- Failure probability prediction
- Remaining Useful Life (RUL) estimation
- Root cause analysis

## Tech Stack
Python, Pandas, Scikit-learn, Streamlit

## Key Features
- Real-time health monitoring
- Failure probability prediction
- RUL estimation
- Root cause analysis
- Interactive dashboard for operational insights

## Example Output
- Health Score: 45 (**CRITICAL**)
- Failure Probability: 68%
- Estimated RUL: 10 hours

## Results
- Failure Probability: 72%
- RUL: 12 hours
- Status: **CRITICAL**

## How to Run
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Project Structure
- `data/sensor_data.csv` — input sensor dataset
- `dashboard.py` — Streamlit app and visualization
- `main.py` — core prediction workflow
- `features.py` — feature engineering logic
- `model.py` — model training/inference utilities
- `utils.py` — helper functions
- `generate_data.py` — sample/simulated data generation

## Disclaimer
This project is intended for educational/prototyping purposes.  
Predictions should support (not replace) engineering judgment in production environments.
