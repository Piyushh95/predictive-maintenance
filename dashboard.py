import os
import importlib
import pandas as pd
import streamlit as st
from datetime import datetime
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
    if not os.path.exists("data/sensor_data.csv"):
        os.makedirs("data", exist_ok=True)
        importlib.import_module("generate_data")

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
    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        page_icon="🏭",
        layout="wide",
    )
    st.markdown(
        """
        <div style="
            padding: 1rem 1.1rem;
            margin-bottom: 0.65rem;
            border: 1px solid #2b2f36;
            border-radius: 14px;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        ">
            <div style="
                font-size: 0.78rem;
                letter-spacing: 0.11em;
                text-transform: uppercase;
                color: #9ca3af;
                margin-bottom: 0.35rem;
            ">Predictive Maintenance Setup</div>
            <div style="
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.2;
                color: #f9fafb;
            ">🏭 Predictive Maintenance Control Center</div>
            <div style="
                margin-top: 0.35rem;
                opacity: 0.88;
                color: #d1d5db;
            ">Operational view for machine health, alerts, and failure risk.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Product-like visual styling for key panels.
    st.markdown(
        """
        <style>
        .app-chip {
            display: inline-block;
            border: 1px solid #2b2f36;
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            margin-right: 0.4rem;
            font-size: 0.8rem;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #2b2f36;
        }
        .sidebar-brand {
            border: 1px solid #2b2f36;
            border-radius: 12px;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            padding: 0.75rem 0.8rem;
            margin-bottom: 0.8rem;
        }
        .status-card {
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            border: 1px solid #2b2f36;
            background: #161b22;
        }
        .status-title {
            font-size: 0.85rem;
            opacity: 0.8;
            margin-bottom: 0.25rem;
        }
        .status-value {
            font-size: 1.35rem;
            font-weight: 700;
        }
        .status-normal { color: #2ecc71; }
        .status-warning { color: #f1c40f; }
        .status-critical, .status-emergency { color: #e74c3c; }
        .kpi-card {
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            border: 1px solid #2b2f36;
            background: #161b22;
            min-height: 86px;
        }
        .kpi-title {
            font-size: 0.82rem;
            opacity: 0.8;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.35rem;
            font-weight: 700;
        }
        .kpi-normal { border-left: 4px solid #2ecc71; }
        .kpi-warning { border-left: 4px solid #f1c40f; }
        .kpi-critical, .kpi-emergency { border-left: 4px solid #e74c3c; }
        .mini-card {
            border: 1px solid #2b2f36;
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            background: #161b22;
            margin-bottom: 0.5rem;
        }
        .mini-title { font-size: 0.78rem; opacity: 0.8; }
        .mini-value { font-size: 1.1rem; font-weight: 700; }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            border: 1px solid #2b2f36;
            padding: 0.3rem 0.8rem;
            background: #161b22;
        }
        .stTabs [aria-selected="true"] {
            background: #1f2937;
            border-color: #3b82f6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <span class="app-chip">Live Monitoring</span>
        <span class="app-chip">Failure Prediction</span>
        <span class="app-chip">Root Cause Insights</span>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls.
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div style="font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af;">
                Monitoring App
            </div>
            <div style="font-size:1.05rem; font-weight:700; margin-top:0.2rem;">Machine Health Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.header("Control Panel")
    if st.sidebar.button("Refresh now"):
        st.rerun()
    auto_refresh = st.sidebar.selectbox(
        "Auto-refresh interval",
        options=["Off", "15 sec", "30 sec", "60 sec"],
        index=0,
    )
    refresh_seconds = {
        "Off": 0,
        "15 sec": 15,
        "30 sec": 30,
        "60 sec": 60,
    }[auto_refresh]
    if refresh_seconds > 0:
        st.sidebar.caption(f"Auto-refreshing every {refresh_seconds} seconds.")

    min_time = feat_df["timestamp"].min()
    max_time = feat_df["timestamp"].max()
    selected_range = st.sidebar.slider(
        "Time range",
        min_value=min_time.to_pydatetime(),
        max_value=max_time.to_pydatetime(),
        value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
        format="YYYY-MM-DD HH:mm",
    )
    alert_options = ["Normal", "Warning", "Critical", "Emergency"]
    selected_alerts = st.sidebar.multiselect(
        "Alert levels to display",
        options=alert_options,
        default=alert_options,
    )

    filtered = feat_df[
        (feat_df["timestamp"] >= pd.Timestamp(selected_range[0]))
        & (feat_df["timestamp"] <= pd.Timestamp(selected_range[1]))
        & (feat_df["alert_level"].isin(selected_alerts))
    ]
    if filtered.empty:
        st.warning("No data points match current filters. Reset filters from the sidebar.")
        return

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Last updated: {now_text}")

    status_class = str(machine_status).strip().lower()
    status_icon = {
        "normal": "🟢",
        "warning": "🟡",
        "critical": "🟠",
        "emergency": "🔴",
    }.get(status_class, "⚪")

    # Top summary metrics (card style)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""
        <div class="kpi-card kpi-{status_class}">
            <div class="kpi-title">Latest Health</div>
            <div class="kpi-value">{filtered['health'].iloc[-1]:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="kpi-card kpi-{status_class}">
            <div class="kpi-title">Machine Status</div>
            <div class="kpi-value">{str(machine_status).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class="kpi-card kpi-{status_class}">
            <div class="kpi-title">Failure Probability</div>
            <div class="kpi-value">{failure_prob * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rul_value = f"{rul_info['remaining_hours']:.1f} h" if rul_info is not None else "N/A"
    col4.markdown(
        f"""
        <div class="kpi-card kpi-{status_class}">
            <div class="kpi-title">RUL (hours)</div>
            <div class="kpi-value">{rul_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">Current Risk Status</div>
            <div class="status-value status-{status_class}">{status_icon} {str(machine_status).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_diagnostics, tab_reports = st.tabs(
        ["Overview", "Diagnostics", "Reports"]
    )

    with tab_overview:
        overview_col1, overview_col2, overview_col3 = st.columns(3)
        overview_col1.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-title">Filtered Data Points</div>
                <div class="mini-value">{len(filtered)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        overview_col2.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-title">Time Window Start</div>
                <div class="mini-value">{filtered['timestamp'].min().strftime("%Y-%m-%d %H:%M")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        overview_col3.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-title">Time Window End</div>
                <div class="mini-value">{filtered['timestamp'].max().strftime("%Y-%m-%d %H:%M")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Health Over Time")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            filtered["timestamp"],
            filtered["health"],
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
            mask = filtered["alert_level"] == level
            ax.scatter(
                filtered.loc[mask, "timestamp"],
                filtered.loc[mask, "health"],
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
        st.pyplot(fig, width="stretch")

    with tab_diagnostics:
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

        st.subheader("Root Cause Analysis (Feature Importances)")
        if feature_importance:
            fi_series = pd.Series(feature_importance).sort_values(ascending=False)
            st.bar_chart(fi_series)
        else:
            st.write("Not enough data or features to compute meaningful feature importances.")

    with tab_reports:
        st.subheader("Failure Report")
        with st.container(border=True):
            st.markdown("Generate a detailed machine risk report and download it as a text file.")
            action_col, info_col = st.columns([1, 2])
            with action_col:
                generate_clicked = st.button(
                    "Generate Failure Report",
                    type="primary",
                    use_container_width=True,
                    disabled=rul_info is None,
                )
            with info_col:
                if rul_info is None:
                    st.info("Time-based RUL is unavailable, so report generation is currently disabled.")
                else:
                    st.success("RUL data is available. Report can be generated.")

            if "report_text" not in st.session_state:
                st.session_state["report_text"] = None

            if generate_clicked and rul_info is not None:
                generate_failure_report(df, feat_df, rul_info, output_path="machine_report.txt")
                with open("machine_report.txt", "r", encoding="utf-8") as f:
                    st.session_state["report_text"] = f.read()
                st.toast("Failure report generated successfully.")

            report_text = st.session_state.get("report_text")
            if report_text:
                preview_col, download_col = st.columns([3, 1])
                with preview_col:
                    st.text_area("Report Preview", report_text, height=280)
                with download_col:
                    st.download_button(
                        label="Download Report",
                        data=report_text,
                        file_name="machine_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
            else:
                st.caption("No report generated in this session yet.")

    if refresh_seconds > 0:
        st.markdown(
            f"""
            <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_seconds * 1000});
            </script>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

