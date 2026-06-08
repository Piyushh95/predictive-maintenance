import os
import importlib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from fleet_manager import FleetManager
from utils import generate_failure_report


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def get_fleet_manager() -> FleetManager:
    """
    NASA CMAPSS (FD001) multi-engine fleet, cached in Streamlit session state.
    """
    if "fleet_mgr" not in st.session_state:
        st.session_state["fleet_mgr"] = FleetManager.from_cmapss(project_root=_project_root())
    return st.session_state["fleet_mgr"]


def run_pipeline_csv_fallback():
    """
    Legacy single-file path using ``data/sensor_data.csv`` (synthetic generator).
    """
    if not os.path.exists("data/sensor_data.csv"):
        os.makedirs("data", exist_ok=True)
        importlib.import_module("generate_data")

    from sklearn.preprocessing import StandardScaler

    from features import create_rolling_features
    from model import train_model, anomaly_score
    from utils import (
        anomaly_to_health,
        assign_alert_level,
        compute_degradation_signals,
        detect_degradation_start,
        estimate_rul,
        get_alert_level,
        predict_failure_probability,
        predict_rul,
        root_cause_analysis,
        train_failure_model,
    )

    df = pd.read_csv("data/sensor_data.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    feat_df = create_rolling_features(df)
    X = feat_df.drop(columns=["timestamp"])
    split = int(0.3 * len(X))
    scaler = StandardScaler()
    scaler.fit(X.iloc[:split])
    X_scaled = scaler.transform(X)
    X_train = X_scaled[:split]
    model = train_model(X_train)
    scores = anomaly_score(model, X_scaled)
    feat_df["anomaly_score"] = scores
    min_s, max_s = scores.min(), scores.max()
    feat_df["health"] = anomaly_to_health(scores, min_s, max_s)
    health_scores_train = feat_df["health"].iloc[:split].values
    failure_model = train_failure_model(X_train, health_scores_train)
    latest_feature_vector = X_scaled[-1]
    failure_prob = predict_failure_probability(failure_model, latest_feature_vector)
    signals = compute_degradation_signals(feat_df["health"])
    feat_df = pd.concat([feat_df, signals], axis=1)
    feat_df["alert_level"] = assign_alert_level(
        feat_df["health"],
        degradation_trend=feat_df["degradation_trend"],
        very_fast_degradation=feat_df["very_fast_degradation"],
    )
    try:
        rul_info = predict_rul(feat_df)
    except Exception:
        rul_info = None
    degr_start_idx = detect_degradation_start(feat_df["health"].values)
    rul_steps = estimate_rul(feat_df["health"].values, threshold=30)
    current_health = float(feat_df["health"].iloc[-1])
    machine_status = get_alert_level(current_health)
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


# ─────────────────────────── CSS ───────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiselect label,
[data-testid="stSidebar"] .stSlider label {
    font-size: 0.82rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Header banner ── */
.dash-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1c2128 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.dash-header-icon {
    font-size: 2.4rem;
    line-height: 1;
}
.dash-header-tag {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #58a6ff;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.dash-header-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #f0f6fc;
    line-height: 1.15;
}
.dash-header-sub {
    font-size: 0.88rem;
    color: #8b949e;
    margin-top: 0.3rem;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
}
.sidebar-brand-tag {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #58a6ff;
    font-weight: 600;
}
.sidebar-brand-name {
    font-size: 1rem;
    font-weight: 700;
    color: #f0f6fc;
    margin-top: 0.15rem;
}

/* ── KPI cards ── */
.kpi-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.kpi-card {
    flex: 1;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-normal::before  { background: linear-gradient(90deg, #2ea043, #3fb950); }
.kpi-warning::before { background: linear-gradient(90deg, #9e6a03, #d29922); }
.kpi-critical::before,
.kpi-emergency::before { background: linear-gradient(90deg, #b91c1c, #ef4444); }

.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.45rem;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1;
    color: #f0f6fc;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 0.3rem;
}
.kpi-icon {
    position: absolute;
    top: 0.85rem;
    right: 1rem;
    font-size: 1.3rem;
    opacity: 0.35;
}

/* ── Status banner ── */
.status-banner {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin: 0.75rem 0 1rem;
}
.status-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-normal   { background: #3fb950; box-shadow: 0 0 6px #3fb95066; }
.dot-warning  { background: #d29922; box-shadow: 0 0 6px #d2992266; }
.dot-critical,
.dot-emergency { background: #ef4444; box-shadow: 0 0 6px #ef444466; }
.status-label {
    font-size: 0.78rem;
    color: #8b949e;
    font-weight: 500;
}
.status-value {
    font-size: 1rem;
    font-weight: 700;
    color: #f0f6fc;
}
.status-normal   .status-value { color: #3fb950; }
.status-warning  .status-value { color: #d29922; }
.status-critical .status-value,
.status-emergency .status-value { color: #ef4444; }

/* ── Mini stat cards (overview tab) ── */
.mini-stat {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.mini-stat-label { font-size: 0.73rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; }
.mini-stat-value { font-size: 1.15rem; font-weight: 700; color: #f0f6fc; margin-top: 0.2rem; }

/* ── Diag stat cards ── */
.diag-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 0.9rem 1rem;
}
.diag-label { font-size: 0.73rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; margin-bottom: 0.3rem; }
.diag-value { font-size: 1.1rem; font-weight: 700; color: #58a6ff; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.3rem; border-bottom: 1px solid #21262d; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    border: 1px solid transparent;
    padding: 0.4rem 1rem;
    font-size: 0.88rem;
    font-weight: 500;
    color: #8b949e;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #161b22 !important;
    border-color: #21262d !important;
    border-bottom-color: #161b22 !important;
    color: #f0f6fc !important;
}

/* ── Fleet table ── */
.fleet-section {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
}
.fleet-section-title {
    font-size: 0.78rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

/* ── Chips ── */
.chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: #8b949e;
}
.chip-dot { width: 6px; height: 6px; border-radius: 50%; background: #58a6ff; }

/* ── Report section ── */
.report-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
}
.report-title {
    font-size: 0.78rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.report-desc { font-size: 0.88rem; color: #8b949e; margin-bottom: 1rem; }

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1rem 0;
}
</style>
"""


def render_kpi_cards(health_val, machine_status, failure_prob, rul_info):
    status_class = str(machine_status).strip().lower()
    icon_map = {"normal": "💚", "warning": "⚠️", "critical": "🔶", "emergency": "🔴"}
    icon = icon_map.get(status_class, "⚪")
    rul_value = f"{rul_info['remaining_hours']:.1f} h" if rul_info is not None else "N/A"

    st.markdown(
        f"""
        <div style="display:flex; gap:0.75rem; margin-bottom:1rem;">
            <div class="kpi-card kpi-{status_class}" style="flex:1">
                <div class="kpi-icon">🩺</div>
                <div class="kpi-label">Health Score</div>
                <div class="kpi-value">{health_val:.1f}</div>
                <div class="kpi-sub">Out of 100</div>
            </div>
            <div class="kpi-card kpi-{status_class}" style="flex:1">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">Machine Status</div>
                <div class="kpi-value">{str(machine_status).upper()}</div>
                <div class="kpi-sub">Current risk level</div>
            </div>
            <div class="kpi-card kpi-{status_class}" style="flex:1">
                <div class="kpi-icon">📊</div>
                <div class="kpi-label">Failure Probability</div>
                <div class="kpi-value">{failure_prob * 100:.1f}%</div>
                <div class="kpi-sub">Next failure likelihood</div>
            </div>
            <div class="kpi-card kpi-{status_class}" style="flex:1">
                <div class="kpi-icon">⏱️</div>
                <div class="kpi-label">Remaining Useful Life</div>
                <div class="kpi-value">{rul_value}</div>
                <div class="kpi-sub">Estimated hours left</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(machine_status):
    status_class = str(machine_status).strip().lower()
    descriptions = {
        "normal": "All systems operating within normal parameters.",
        "warning": "Degradation detected — increased monitoring recommended.",
        "critical": "Significant degradation — schedule maintenance soon.",
        "emergency": "Imminent failure risk — immediate action required!",
    }
    desc = descriptions.get(status_class, "Status unknown.")
    st.markdown(
        f"""
        <div class="status-banner status-{status_class}">
            <div class="status-dot dot-{status_class}"></div>
            <div>
                <div class="status-label">Current Risk Status</div>
                <div class="status-value">{str(machine_status).upper()} — {desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_health_chart(filtered):
    level_colors = {
        "Normal": "#3fb950",
        "Warning": "#d29922",
        "Critical": "#f97316",
        "Emergency": "#ef4444",
    }

    fig = go.Figure()

    # Base health line
    fig.add_trace(
        go.Scatter(
            x=filtered["timestamp"],
            y=filtered["health"],
            mode="lines",
            name="Health Score",
            line=dict(color="#30363d", width=1.5),
            showlegend=True,
        )
    )

    # Scatter per alert level
    for level, color in level_colors.items():
        mask = filtered["alert_level"] == level
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=filtered.loc[mask, "timestamp"],
                    y=filtered.loc[mask, "health"],
                    mode="markers",
                    name=level,
                    marker=dict(color=color, size=5 if level == "Normal" else 7, opacity=0.85),
                )
            )

    # Threshold lines
    for y_val, color, label in [
        (70, "#d29922", "Warning threshold"),
        (50, "#f97316", "Critical threshold"),
        (30, "#ef4444", "Emergency threshold"),
    ]:
        fig.add_hline(
            y=y_val,
            line_dash="dash",
            line_color=color,
            line_width=1,
            annotation_text=label,
            annotation_position="left",
            annotation_font_size=11,
            annotation_font_color=color,
        )

    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, sans-serif", color="#8b949e", size=12),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="#21262d", linecolor="#21262d", tickcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#21262d", tickcolor="#21262d", title="Health Score"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_rca_chart(feature_importance):
    fi_series = pd.Series(feature_importance).sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=fi_series.values,
            y=fi_series.index,
            orientation="h",
            marker=dict(
                color=fi_series.values,
                colorscale=[[0, "#21262d"], [0.5, "#388bfd"], [1, "#58a6ff"]],
                showscale=False,
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, sans-serif", color="#8b949e", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#21262d", linecolor="#21262d", title="Importance"),
        yaxis=dict(gridcolor="#21262d", linecolor="#21262d"),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        page_icon="🏭",
        layout="wide",
    )
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="dash-header">
            <div class="dash-header-icon">🏭</div>
            <div>
                <div class="dash-header-tag">Predictive Maintenance</div>
                <div class="dash-header-title">Dashboard</div>
                <div class="dash-header-sub">NASA CMAPSS (PHM08 FD001) fleet · Real-time health, alerts &amp; RUL monitoring</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-tag">Machine Health</div>
            <div class="sidebar-brand-name">Control Panel</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset = st.sidebar.radio(
        "Dataset",
        ["NASA CMAPSS (FD001)", "Local CSV (synthetic)"],
        index=0,
        help="CMAPSS sensors s4/s11/s7 are mapped to vibration/temperature/pressure.",
    )
    fleet = None
    machine_id = None
    if dataset.startswith("NASA"):
        fleet = get_fleet_manager()
        machine_id = st.sidebar.selectbox(
            "Engine unit",
            fleet.trained_machine_ids(),
            index=0,
        )

    st.sidebar.markdown("<hr style='border-color:#21262d; margin:0.75rem 0;'>", unsafe_allow_html=True)

    col_refresh, col_auto = st.sidebar.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col_auto:
        auto_refresh = st.selectbox(
            "Auto",
            options=["Off", "15s", "30s", "60s"],
            index=0,
            label_visibility="collapsed",
        )

    refresh_seconds = {"Off": 0, "15s": 15, "30s": 30, "60s": 60}[auto_refresh]
    if refresh_seconds > 0:
        st.sidebar.caption(f"⏱ Auto-refreshing every {refresh_seconds}s")

    st.sidebar.markdown("<hr style='border-color:#21262d; margin:0.75rem 0;'>", unsafe_allow_html=True)

    # ── Pipeline ──────────────────────────────────────────────
    with st.spinner("Loading predictive maintenance pipeline…"):
        if fleet is not None:
            results = fleet.dashboard_run_for_machine(machine_id)
        else:
            results = run_pipeline_csv_fallback()

    df = results["df"]
    feat_df = results["feat_df"]
    failure_prob = results["failure_prob"]
    rul_info = results["rul_info"]
    degr_start_idx = results["degr_start_idx"]
    rul_steps = results["rul_steps"]
    machine_status = results["machine_status"]
    feature_importance = results["feature_importance"]

    # ── Sidebar time & alert filters ─────────────────────────
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
        "Alert levels",
        options=alert_options,
        default=alert_options,
    )

    # ── Fleet overview ─────────────────────────────────────────
    if fleet is not None:
        st.markdown('<div class="fleet-section">', unsafe_allow_html=True)
        st.markdown('<div class="fleet-section-title">🛩 Fleet Overview — Trained Engines</div>', unsafe_allow_html=True)
        fleet_df = pd.DataFrame(fleet.fleet_summary())
        st.dataframe(
            fleet_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Charts below refer to engine unit **{machine_id}**.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Filter data ────────────────────────────────────────────
    filtered = feat_df[
        (feat_df["timestamp"] >= pd.Timestamp(selected_range[0]))
        & (feat_df["timestamp"] <= pd.Timestamp(selected_range[1]))
        & (feat_df["alert_level"].isin(selected_alerts))
    ]
    if filtered.empty:
        st.warning("⚠️ No data points match the current filters. Reset filters from the sidebar.")
        return

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Chips row ──────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="chip-row">
            <span class="chip"><span class="chip-dot"></span>Live Monitoring</span>
            <span class="chip"><span class="chip-dot"></span>Failure Prediction</span>
            <span class="chip"><span class="chip-dot"></span>Root Cause Insights</span>
            <span class="chip" style="margin-left:auto; color:#58a6ff; border-color:#388bfd33; background:#1c2128;">
                🕐 Updated {now_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI cards ──────────────────────────────────────────────
    current_health = float(filtered["health"].iloc[-1])
    render_kpi_cards(current_health, machine_status, failure_prob, rul_info)

    # ── Status banner ──────────────────────────────────────────
    render_status_banner(machine_status)

    # ── Tabs ────────────────────────────────────────────────────
    tab_overview, tab_diagnostics, tab_reports = st.tabs(
        ["📈  Overview", "🔬  Diagnostics", "📄  Reports"]
    )

    # ── OVERVIEW TAB ──────────────────────────────────────────
    with tab_overview:
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="mini-stat"><div class="mini-stat-label">Data Points (filtered)</div>'
            f'<div class="mini-stat-value">{len(filtered):,}</div></div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="mini-stat"><div class="mini-stat-label">Time Window Start</div>'
            f'<div class="mini-stat-value">{filtered["timestamp"].min().strftime("%Y-%m-%d %H:%M")}</div></div>',
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="mini-stat"><div class="mini-stat-label">Time Window End</div>'
            f'<div class="mini-stat-value">{filtered["timestamp"].max().strftime("%Y-%m-%d %H:%M")}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(
            '<span style="font-size:0.78rem; font-weight:700; letter-spacing:0.08em; '
            'text-transform:uppercase; color:#58a6ff;">Health Score Over Time</span>',
            unsafe_allow_html=True,
        )
        render_health_chart(filtered)

    # ── DIAGNOSTICS TAB ───────────────────────────────────────
    with tab_diagnostics:
        st.markdown(
            '<span style="font-size:0.78rem; font-weight:700; letter-spacing:0.08em; '
            'text-transform:uppercase; color:#58a6ff;">Degradation &amp; RUL Diagnostics</span>',
            unsafe_allow_html=True,
        )
        degr_text = str(degr_start_idx) if degr_start_idx is not None else "Not detected"
        rul_steps_text = str(rul_steps) if rul_steps is not None else "N/A"
        pred_fail_text = str(rul_info["predicted_failure_time"]) if rul_info is not None else "N/A"

        d1, d2, d3 = st.columns(3)
        d1.markdown(
            f'<div class="diag-card"><div class="diag-label">Degradation Start Index</div>'
            f'<div class="diag-value">{degr_text}</div></div>',
            unsafe_allow_html=True,
        )
        d2.markdown(
            f'<div class="diag-card"><div class="diag-label">Estimated RUL (timesteps to health 30)</div>'
            f'<div class="diag-value">{rul_steps_text}</div></div>',
            unsafe_allow_html=True,
        )
        d3.markdown(
            f'<div class="diag-card"><div class="diag-label">Predicted Failure Time</div>'
            f'<div class="diag-value">{pred_fail_text}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(
            '<span style="font-size:0.78rem; font-weight:700; letter-spacing:0.08em; '
            'text-transform:uppercase; color:#58a6ff;">Root Cause Analysis — Feature Importances</span>',
            unsafe_allow_html=True,
        )
        if feature_importance:
            render_rca_chart(feature_importance)
        else:
            st.info("Not enough data or features to compute meaningful feature importances.")

    # ── REPORTS TAB ───────────────────────────────────────────
    with tab_reports:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="report-title">📄 Failure Report Generator</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="report-desc">Generate a detailed machine risk report based on current sensor data and RUL predictions.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        action_col, info_col = st.columns([1, 2])
        with action_col:
            generate_clicked = st.button(
                "⚙️ Generate Report",
                type="primary",
                use_container_width=True,
                disabled=rul_info is None,
            )
        with info_col:
            if rul_info is None:
                st.info("Time-based RUL is unavailable — report generation is currently disabled.")
            else:
                st.success("✅ RUL data is available. Report can be generated.")

        if "report_text" not in st.session_state:
            st.session_state["report_text"] = None

        if generate_clicked and rul_info is not None:
            generate_failure_report(df, feat_df, rul_info, output_path="machine_report.txt")
            with open("machine_report.txt", "r", encoding="utf-8") as f:
                st.session_state["report_text"] = f.read()
            st.toast("✅ Failure report generated successfully.")

        report_text = st.session_state.get("report_text")
        if report_text:
            preview_col, download_col = st.columns([3, 1])
            with preview_col:
                st.text_area("Report Preview", report_text, height=300)
            with download_col:
                st.download_button(
                    label="⬇️ Download Report",
                    data=report_text,
                    file_name="machine_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.caption("No report generated in this session yet.")

    # ── Auto-refresh script ────────────────────────────────────
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
