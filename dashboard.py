import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from features import create_rolling_features
from model import train_model, anomaly_score
from utils import (
    anomaly_to_health,
    compute_degradation_signals,
    assign_alert_level,
    predict_rul,
    generate_failure_report,
    detect_degradation_start,
    estimate_rul,
    get_alert_level,
    root_cause_analysis,
    train_failure_model,
    predict_failure_probability,
)


def run_pipeline():
    """
    Run the full predictive maintenance pipeline and return all key artifacts.

    This mirrors the logic in main.py but is structured as a pure function to
    make integration with a web UI straightforward.
    """
    # Load and prepare data
    df = pd.read_csv("data/sensor_data.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    # Feature engineering
    feat_df = create_rolling_features(df)

    X = feat_df.drop(columns=["timestamp"])

    # Train on early normal data (first 30%)
    split = int(0.3 * len(X))

    scaler = StandardScaler()
    scaler.fit(X.iloc[:split])

    X_scaled = scaler.transform(X)
    X_train = X_scaled[:split]

    model = train_model(X_train)

    # Anomaly scoring
    scores = anomaly_score(model, X_scaled)
    feat_df["anomaly_score"] = scores

    # Health score
    min_s, max_s = scores.min(), scores.max()
    feat_df["health"] = anomaly_to_health(scores, min_s, max_s)

    # Failure probability model
    health_scores_train = feat_df["health"].iloc[:split].values
    failure_model = train_failure_model(X_train, health_scores_train)
    latest_feature_vector = X_scaled[-1]
    failure_prob = predict_failure_probability(failure_model, latest_feature_vector)

    # Degradation + alert escalation
    signals = compute_degradation_signals(feat_df["health"])
    feat_df = pd.concat([feat_df, signals], axis=1)
    feat_df["alert_level"] = assign_alert_level(
        feat_df["health"],
        degradation_trend=feat_df["degradation_trend"],
        very_fast_degradation=feat_df["very_fast_degradation"],
    )

    # Time-based RUL + failure report info
    try:
        rul_info = predict_rul(feat_df)
    except Exception:
        rul_info = None

    # Additional RUL diagnostics (timesteps version)
    degr_start_idx = detect_degradation_start(feat_df["health"].values)
    rul_steps = estimate_rul(feat_df["health"].values, threshold=30)

    current_health = float(feat_df["health"].iloc[-1])
    machine_status = get_alert_level(current_health)

    # Root-cause feature importance (RandomForest)
    failure_mask = feat_df["alert_level"].isin(
        ["Critical", "Emergency", "CRITICAL", "EMERGENCY"]
    )
    y_rca = failure_mask.astype(int)
    sensor_feature_cols = [
        c
        for c in feat_df.columns
        if any(c.startswith(prefix) for prefix in ("vibration", "temperature", "pressure"))
    ]
    feature_importance = {}
    if sensor_feature_cols:
        X_rca = feat_df[sensor_feature_cols]
        feature_importance = root_cause_analysis(X_rca, y_rca)

    return {
        "df": df,
        "feat_df": feat_df,
        "failure_prob": failure_prob,
        "rul_info": rul_info,
        "degr_start_idx": degr_start_idx,
        "rul_steps": rul_steps,
        "machine_status": machine_status,
        "feature_importance": feature_importance,
    }


def main():
    st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
    st.title("Predictive Maintenance Dashboard")

    with st.spinner("Running predictive maintenance pipeline..."):
        results = run_pipeline()

    df = results["df"]
    feat_df = results["feat_df"]
    failure_prob = results["failure_prob"]
    rul_info = results["rul_info"]
    degr_start_idx = results["degr_start_idx"]
    rul_steps = results["rul_steps"]
    machine_status = results["machine_status"]
    feature_importance = results["feature_importance"]

    # Top summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Health", f"{feat_df['health'].iloc[-1]:.1f}")
    col2.metric("Machine Status", machine_status)
    col3.metric("Failure Probability", f"{failure_prob * 100:.1f}%")
    if rul_info is not None:
        col4.metric("RUL (hours)", f"{rul_info['remaining_hours']:.1f}")
    else:
        col4.metric("RUL (hours)", "N/A")

    # Health over time plot with alert levels
    st.subheader("Health Over Time")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        feat_df["timestamp"],
        feat_df["health"],
        label="Health Score",
        color="grey",
        alpha=0.5,
    )

    level_colors = {
        "Normal": "blue",
        "Warning": "yellow",
        "Critical": "orange",
        "Emergency": "red",
    }

    for level, color in level_colors.items():
        mask = feat_df["alert_level"] == level
        ax.scatter(
            feat_df.loc[mask, "timestamp"],
            feat_df.loc[mask, "health"],
            color=color,
            s=10 if level != "Normal" else 6,
            alpha=0.9 if level != "Normal" else 0.6,
            label=level,
        )

    ax.axhline(70, color="yellow", linestyle="--", linewidth=1)
    ax.axhline(50, color="orange", linestyle="--", linewidth=1)
    ax.axhline(30, color="red", linestyle="--", linewidth=1)
    ax.set_title("Machine Health Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Health")
    ax.legend(loc="lower left", fontsize="small", ncol=2)
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)

    # Diagnostic information
    st.subheader("Degradation & RUL Diagnostics")
    cols = st.columns(3)
    degr_text = (
        str(degr_start_idx) if degr_start_idx is not None else "Not detected"
    )
    cols[0].write(f"Degradation start index (rolling slope): **{degr_text}**")
    if rul_steps is not None:
        cols[1].write(f"Estimated RUL (timesteps to health 30): **{rul_steps}**")
    else:
        cols[1].write("Estimated RUL (timesteps): **N/A**")
    if rul_info is not None:
        cols[2].write(
            f"Predicted failure time: **{rul_info['predicted_failure_time']}**"
        )
    else:
        cols[2].write("Predicted failure time: **N/A**")

    # Root cause feature importance
    st.subheader("Root Cause Analysis (Feature Importances)")
    if feature_importance:
        fi_series = pd.Series(feature_importance).sort_values(ascending=False)
        st.bar_chart(fi_series)
    else:
        st.write("Not enough data or features to compute meaningful feature importances.")

    # Failure report generation / download
    st.subheader("Failure Report")
    if st.button("Generate failure report"):
        if rul_info is None:
            st.warning("Cannot generate failure report: time-based RUL is not available.")
        else:
            generate_failure_report(df, feat_df, rul_info, output_path="machine_report.txt")
            with open("machine_report.txt", "r", encoding="utf-8") as f:
                report_text = f.read()
            st.text_area("Report preview", report_text, height=300)
            st.download_button(
                label="Download report",
                data=report_text,
                file_name="machine_report.txt",
                mime="text/plain",
            )
    else:
        st.caption("Click the button above to generate a fresh failure report.")


if __name__ == "__main__":
    main()

