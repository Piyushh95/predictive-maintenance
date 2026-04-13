import numpy as np

def anomaly_to_health(score, min_s, max_s):
    if max_s == min_s:
        return np.full_like(score, 100.0)
    # Normalize anomaly score → health (0–100)
    score = np.clip(score, min_s, max_s)
    health = 100 * (1 - (score - min_s) / (max_s - min_s))
    return health

def detect_degradation_start(health_scores, window: int = 10, slope_threshold: float = -0.5):
    """
    Detect the start of degradation using a rolling linear slope on health scores.

    A degradation start is flagged when the slope of a window of `window` points
    is smaller than `slope_threshold` (e.g. -0.5 health points per step).

    Returns the index (integer position into `health_scores`) of the first
    detected degradation window, or None if no degradation is found.
    """
    import numpy as np

    hs = np.asarray(health_scores, dtype=float)
    if len(hs) < window:
        return None

    slopes = []
    x = np.arange(window, dtype=float)
    for i in range(len(hs) - window):
        y = hs[i : i + window]
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)

    for i, s in enumerate(slopes):
        if s < slope_threshold:
            return i
    return None

def estimate_rul(health_scores, threshold: float = 30.0, tail_window: int = 20):
    """
    Estimate Remaining Useful Life (RUL) in timesteps until health reaches `threshold`.

    This uses a simple average degradation rate over the last `tail_window`
    points. It is intentionally lightweight and easy to explain for portfolio use.

    Returns an integer number of timesteps, or None if degradation is not detected
    (non-negative average rate).
    """
    import numpy as np

    hs = np.asarray(health_scores, dtype=float)
    if len(hs) < 2:
        return None

    current = hs[-1]

    # If we are already at or below the failure threshold, there is no
    # remaining useful life in timesteps.
    if current <= threshold:
        return 0

    tail = hs[-tail_window:] if len(hs) >= tail_window else hs
    degradation_rate = np.mean(np.diff(tail))

    if degradation_rate >= 0:
        return None

    rul = (threshold - current) / degradation_rate
    if np.isnan(rul) or np.isinf(rul):
        return None
    # If the linear projection says we've already crossed or are exactly at
    # the threshold, clamp RUL to zero instead of returning a misleading
    # positive value.
    if rul <= 0:
        return 0
    return int(rul)

def get_alert_level(score: float) -> str:
    """
    Map a single health score to a 4-level alert string.

    This is a lightweight, per-sample classifier that complements the
    time-series-based `assign_alert_level` function.
    """
    if score < 30:
        return "EMERGENCY"
    elif score < 50:
        return "CRITICAL"
    elif score < 70:
        return "WARNING"
    else:
        return "NORMAL"

def predict_rul(
    df,
    *,
    timestamp_col: str = "timestamp",
    health_col: str = "health",
    consecutive_decrease: int = 5,
    smooth_window: int = 3,
    failure_threshold: float = 20.0,
):
    """
    Predict Remaining Useful Life (RUL) using a simple linear health trend model.

    Steps implemented (per project spec):
    1) Detect degradation start (strict drops on lightly smoothed health, or rolling-slope fallback).
    2) Fit LinearRegression on timestamp vs health (from degradation start onward).
    3) Predict when health will reach failure threshold (health = 20).
    4) Return predicted failure time, remaining hours and degradation rate.

    Notes:
    - This function expects a DataFrame that already contains `timestamp` and `health`.
      In this project, `main.py` computes health from anomaly scores, then calls this.
    - We model time as hours elapsed since degradation start to keep numbers stable.
    - Streak detection uses a short rolling mean to tolerate single-tick noise; if no
      streak is found, `detect_degradation_start` is used as a fallback start index.
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    if timestamp_col not in df.columns or health_col not in df.columns:
        raise ValueError(
            f"predict_rul requires columns {timestamp_col!r} and {health_col!r}"
        )

    work = df[[timestamp_col, health_col]].copy()
    work = work.dropna(subset=[timestamp_col, health_col])
    work = work.sort_values(timestamp_col)

    if len(work) < (consecutive_decrease + 2):
        raise ValueError("Not enough data to detect degradation and fit RUL model.")

    # 1) Degradation start: N consecutive decreases on smoothed health (reduces noise).
    health_raw = pd.Series(work[health_col].astype(float).to_numpy(), index=work.index)
    if smooth_window and smooth_window > 1:
        health_for_streak = health_raw.rolling(smooth_window, min_periods=1).mean()
    else:
        health_for_streak = health_raw

    dec = health_for_streak.diff().lt(0)
    streak = dec.rolling(consecutive_decrease, min_periods=consecutive_decrease).sum()
    hit = streak.eq(consecutive_decrease)

    if hit.any():
        end_idx = hit.idxmax()
        pos = work.index.get_loc(end_idx)
        start_pos = max(0, int(pos) - (consecutive_decrease - 1))
    else:
        # Fallback: same rolling-slope idea as detect_degradation_start (aligned to rows).
        slope_idx = detect_degradation_start(health_raw.to_numpy())
        if slope_idx is None:
            raise ValueError(
                "Could not detect degradation start "
                f"(no {consecutive_decrease} consecutive decreases on smoothed health, "
                "and rolling-slope fallback found no window)."
            )
        start_pos = int(slope_idx)

    degr = work.iloc[start_pos:].copy()
    t0 = pd.to_datetime(degr[timestamp_col].iloc[0])
    t_last = pd.to_datetime(degr[timestamp_col].iloc[-1])

    # 2) Fit linear regression: time(hours since t0) -> health.
    t_hours = (pd.to_datetime(degr[timestamp_col]) - t0).dt.total_seconds() / 3600.0
    X = t_hours.to_numpy().reshape(-1, 1)
    y = degr[health_col].astype(float).to_numpy()

    lr = LinearRegression()
    lr.fit(X, y)

    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)

    # If the long window is flat/noisy upward, refit on the recent tail only.
    if slope >= 0 and len(degr) > 15:
        tail_n = max(15, min(50, len(degr) // 2))
        degr = degr.iloc[-tail_n:].copy()
        t0 = pd.to_datetime(degr[timestamp_col].iloc[0])
        t_last = pd.to_datetime(degr[timestamp_col].iloc[-1])
        t_hours = (pd.to_datetime(degr[timestamp_col]) - t0).dt.total_seconds() / 3600.0
        X = t_hours.to_numpy().reshape(-1, 1)
        y = degr[health_col].astype(float).to_numpy()
        lr.fit(X, y)
        slope = float(lr.coef_[0])
        intercept = float(lr.intercept_)

    # 3) Predict when health reaches failure threshold.
    if slope >= 0:
        raise ValueError(
            f"Health is not degrading in fitted window (slope={slope:.6f}); cannot predict failure time."
        )

    x_fail_hours = (failure_threshold - intercept) / slope  # slope is negative

    if x_fail_hours <= float(X[-1, 0]):
        # Already at/under threshold within (or before end of) fitted window.
        predicted_failure_time = pd.to_datetime(t_last).to_pydatetime()
        remaining_hours = 0.0
    else:
        predicted_failure_time = (t0 + pd.to_timedelta(x_fail_hours, unit="h")).to_pydatetime()
        remaining_hours = (predicted_failure_time - t_last.to_pydatetime()).total_seconds() / 3600.0

    # 4) Output
    return {
        "predicted_failure_time": predicted_failure_time,
        "remaining_hours": float(max(0.0, remaining_hours)),
        # Negative slope: health points lost per hour during degradation.
        "degradation_rate_per_hour": slope,
    }

def generate_failure_report(
    raw_df,
    feat_df,
    rul_info,
    output_path: str = "machine_report.txt",
    failure_threshold: float = 20.0,
):
    """
    Generate a human-readable failure report for plant engineers.

    The report summarizes:
    - Timestamp of failure (when health first drops below failure_threshold, if it has)
    - When warning started (first non-Normal alert level)
    - Degradation rate (health points per hour from the RUL model)
    - Predicted failure time and remaining useful life
    - A simple inferred root cause and recommended maintenance action
    """
    import pandas as pd
    import numpy as np

    feat_df = feat_df.copy()

    # Ensure timestamps are in datetime for consistent formatting.
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

    # When did the system first raise any warning? (first non-Normal alert)
    if "alert_level" in feat_df.columns:
        warn_mask = feat_df["alert_level"].isin(["Warning", "Critical", "Emergency"])
        warning_start_time = (
            feat_df.loc[warn_mask, "timestamp"].iloc[0]
            if warn_mask.any()
            else None
        )
    else:
        warning_start_time = None

    # When did the health actually cross the failure threshold?
    if "health" in feat_df.columns:
        fail_mask = feat_df["health"] <= failure_threshold
        failure_time = (
            feat_df.loc[fail_mask, "timestamp"].iloc[0]
            if fail_mask.any()
            else None
        )
    else:
        failure_time = None

    predicted_failure_time = rul_info.get("predicted_failure_time")
    remaining_hours = float(rul_info.get("remaining_hours", float("nan")))
    degradation_rate = float(rul_info.get("degradation_rate_per_hour", float("nan")))

    # Infer a simple root cause by comparing sensor behaviour before and after warnings.
    def _infer_root_cause_and_recommendation():
        sensor_cols = [
            col
            for col in ["vibration", "temperature", "pressure"]
            if col in raw_df.columns
        ]
        if not sensor_cols or warning_start_time is None:
            return (
                "Degradation detected, but sensor pattern is not specific enough to "
                "identify a dominant root cause.",
                "Perform a general inspection of the machine, check lubrication, "
                "mechanical clearances, and verify process conditions.",
            )

        baseline_df = raw_df[raw_df["timestamp"] < warning_start_time]
        degraded_end = failure_time or predicted_failure_time or raw_df["timestamp"].iloc[-1]
        degraded_df = raw_df[
            (raw_df["timestamp"] >= warning_start_time)
            & (raw_df["timestamp"] <= degraded_end)
        ]

        if len(baseline_df) < 20 or len(degraded_df) < 20:
            return (
                "Degradation detected, but there is insufficient baseline/degraded "
                "data to clearly identify the root cause.",
                "Inspect the machine for abnormal noise, temperature, and pressure "
                "behaviour; follow standard troubleshooting procedures.",
            )

        baseline_mean = baseline_df[sensor_cols].mean()
        degraded_mean = degraded_df[sensor_cols].mean()
        delta = degraded_mean - baseline_mean

        # Relative change (percentage-like) to compare across sensors.
        rel_change = delta / (np.abs(baseline_mean) + 1e-6)
        key_sensor = rel_change.abs().idxmax()

        if key_sensor == "vibration":
            root = (
                "Dominant degradation is observed in vibration levels, indicating "
                "possible mechanical imbalance, misalignment, or bearing wear."
            )
            rec = (
                "Inspect rotating components (bearings, couplings, shafts), check "
                "alignment and balance, and verify mounting rigidity and lubrication."
            )
        elif key_sensor == "temperature":
            root = (
                "Dominant degradation is observed in temperature, suggesting "
                "overheating due to friction, poor lubrication, or cooling system issues."
            )
            rec = (
                "Check lubrication quality and quantity, inspect cooling paths "
                "(fans, pumps, heat exchangers), and ensure the machine is not "
                "overloaded."
            )
        elif key_sensor == "pressure":
            root = (
                "Dominant degradation is observed in pressure readings, indicating "
                "possible blockage, leakage, or valve/seal degradation in the process line."
            )
            rec = (
                "Inspect filters and piping for blockage, check for leaks, verify "
                "valve operation and seal integrity, and confirm correct setpoints."
            )
        else:
            root = (
                "Degradation is detected, but no single sensor dominates the change."
            )
            rec = (
                "Perform a general mechanical and process inspection, including "
                "vibration, temperature, and pressure checks."
            )

        return root, rec

    root_cause, recommendation = _infer_root_cause_and_recommendation()

    def _fmt(ts):
        if ts is None:
            return "Not reached yet"
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")

    now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []
    lines.append("MACHINE HEALTH FAILURE REPORT")
    lines.append("=" * 32)
    lines.append(f"Report generated at: {now_str}")
    lines.append("")
    lines.append("Timeline")
    lines.append("--------")
    lines.append(f"- First warning alert:     {_fmt(warning_start_time)}")
    lines.append(f"- Failure threshold (<= {failure_threshold:.1f}) crossed: {_fmt(failure_time)}")
    lines.append(f"- Predicted failure time:  {_fmt(predicted_failure_time)}")
    lines.append("")
    lines.append("Remaining Useful Life")
    lines.append("----------------------")
    if np.isnan(remaining_hours):
        lines.append("- Remaining useful life could not be estimated from the current data.")
    else:
        lines.append(f"- Estimated remaining time: {remaining_hours:0.2f} hours")
    if not np.isnan(degradation_rate):
        lines.append(f"- Average degradation rate: {degradation_rate:0.3f} health points per hour")
    lines.append("")
    lines.append("Diagnosis")
    lines.append("---------")
    lines.append(f"- Root cause (most likely): {root_cause}")
    lines.append(f"- Recommended maintenance:  {recommendation}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def compute_degradation_signals(
    health: "np.ndarray | list | tuple | object",
    *,
    smooth_window: int = 3,
    trend_window: int = 5,
    fast_drop_points: float = 8.0,
    fast_drop_timestamps: int = 5,
):
    """
    Compute degradation-related signals from a health time series.

    This is kept as a standalone utility so alerting logic stays modular and
    easy to adjust without touching model training/scoring code.

    Returns a DataFrame with:
    - health_smooth: rolling-mean smoothed health
    - health_diff: per-timestamp change (diff) of the smoothed health
    - degradation_trend: mostly-decreasing trend over the recent window
    - very_fast_degradation: health drops > `fast_drop_points` within
      `fast_drop_timestamps` timestamps (per project requirement)
    """
    import pandas as pd

    s = pd.Series(health).astype(float)

    # Smoothed health is useful for visualization, but the required
    # "rate of degradation" feature is computed from raw health via diff().
    health_smooth = s.rolling(smooth_window, min_periods=1).mean()

    # Required new feature: rate of degradation from health.diff()
    health_diff = s.diff()

    # "Degradation trend detected" = recent window is mostly decreasing and
    # average change is negative.
    recent_neg = (health_diff < 0).rolling(trend_window, min_periods=1).sum()
    degradation_trend = (health_diff.rolling(trend_window, min_periods=1).mean() < 0) & (
        recent_neg >= max(1, trend_window - 1)
    )

    # New feature: fast degradation if health drops > 8 points within 5 timestamps.
    fast_degradation = s.diff(fast_drop_timestamps) < (-fast_drop_points)

    # Emergency rule mentions "very fast degradation"; we treat it as the same
    # signal defined above (fast drop > 8 within 5 timestamps).
    very_fast_degradation = fast_degradation

    return pd.DataFrame(
        {
            "health_smooth": health_smooth,
            "health_diff": health_diff,
            "degradation_trend": degradation_trend.fillna(False),
            "fast_degradation": fast_degradation.fillna(False),
            "very_fast_degradation": very_fast_degradation.fillna(False),
        }
    )

def assign_alert_level(
    health,
    *,
    degradation_trend=None,
    very_fast_degradation=None,
    warning_threshold: float = 70.0,
    critical_threshold: float = 50.0,
    emergency_threshold: float = 30.0,
):
    """
    Assign a 4-state alert level: Normal → Warning → Critical → Emergency.

    Escalation rules (most severe wins):
    - Warning: health < 70 OR degradation trend detected
    - Critical: health < 50
    - Emergency: health < 30 OR very fast degradation
    """
    import pandas as pd

    h = pd.Series(health).astype(float)
    trend = pd.Series(degradation_trend) if degradation_trend is not None else pd.Series(False, index=h.index)
    vfast = pd.Series(very_fast_degradation) if very_fast_degradation is not None else pd.Series(False, index=h.index)

    level = pd.Series("Normal", index=h.index, dtype="object")

    warning = (h < warning_threshold) | trend.fillna(False)
    critical = h < critical_threshold
    emergency = (h < emergency_threshold) | vfast.fillna(False)

    # Apply in increasing severity; later assignments override earlier ones.
    level.loc[warning] = "Warning"
    level.loc[critical] = "Critical"
    level.loc[emergency] = "Emergency"

    return level

def train_failure_model(X, health_scores, threshold: float = 40.0):
    """
    Train a RandomForest classifier to estimate failure probability.

    A simple binary label is created from health scores:
    - 1 (failure) if health < `threshold`
    - 0 (normal) otherwise

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix used for training (e.g. scaled rolling features).
    health_scores : array-like
        Health scores aligned with rows of X.
    threshold : float, default 40.0
        Health value below which a point is considered a failure example.
    """
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    hs = np.asarray(health_scores, dtype=float)
    y = (hs < threshold).astype(int)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X, y)
    return model

def predict_failure_probability(model, latest_features):
    """
    Predict probability of failure for the latest machine state.

    Parameters
    ----------
    model : fitted classifier
        Model returned by `train_failure_model`.
    latest_features : array-like
        Feature vector describing the latest machine state.

    Returns
    -------
    float
        Probability (0–1) that the latest state is in failure class.
    """
    import numpy as np

    x = np.asarray(latest_features, dtype=float).reshape(1, -1)
    prob = model.predict_proba(x)[0][1]
    return float(prob)

def root_cause_analysis(X, y):
    """
    Train a RandomForestClassifier to quantify which sensor features
    contribute most to failure vs normal states.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features (e.g. vibration / temperature / pressure statistics).
    y : array-like
        Binary target: 1 = failure/alert, 0 = normal.

    Returns
    -------
    dict
        Mapping of feature name → importance (float), sorted descending.
    """
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    if X.empty:
        return {}

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    importances = np.asarray(model.feature_importances_, dtype=float)
    feature_importance = dict(
        sorted(
            zip(X.columns, importances),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    return feature_importance

def analyze_failure_cause(df):
    """
    High-level helper that runs root cause analysis on the current dataset.

    It:
    - Builds a simple failure label from alert level / health.
    - Trains a RandomForestClassifier on vibration / temperature / pressure
      features.
    - Aggregates importances per sensor type and maps them to human-readable
      causes:
        * vibration   → bearing wear
        * temperature → lubrication or cooling issue
        * pressure    → valve or blockage

    Returns
    -------
    (most_likely_cause, feature_contributions)
        most_likely_cause: string description
        feature_contributions: dict of sensor → percentage contribution.
    """
    import pandas as pd
    import numpy as np

    if "health" not in df.columns and "alert_level" not in df.columns:
        raise ValueError("analyze_failure_cause expects 'health' and/or 'alert_level' columns.")

    work = df.copy()

    # Build a binary target: failure vs normal.
    if "alert_level" in work.columns:
        failure_mask = work["alert_level"].isin(["Critical", "Emergency", "CRITICAL", "EMERGENCY"])
    else:
        failure_mask = work["health"] < 30

    y = failure_mask.astype(int)

    sensor_prefixes = {
        "vibration": "bearing wear",
        "temperature": "lubrication or cooling issue",
        "pressure": "valve or blockage",
    }

    feature_cols = [
        c
        for c in work.columns
        if any(c.startswith(prefix) for prefix in sensor_prefixes.keys())
    ]

    if not feature_cols:
        raise ValueError("No sensor feature columns found for root cause analysis.")

    X = work[feature_cols]
    importance_per_feature = root_cause_analysis(X, y)

    # Aggregate feature importances by sensor family.
    sensor_contrib = {k: 0.0 for k in sensor_prefixes.keys()}
    for feat, imp in importance_per_feature.items():
        for sensor in sensor_contrib.keys():
            if feat.startswith(sensor):
                sensor_contrib[sensor] += float(imp)
                break

    total = sum(sensor_contrib.values())
    if total <= 0:
        feature_contributions = {k: 0.0 for k in sensor_contrib.keys()}
    else:
        feature_contributions = {k: (v / total) * 100.0 for k, v in sensor_contrib.items()}

    # Pick most likely cause from highest-contributing sensor.
    top_sensor = max(feature_contributions.items(), key=lambda kv: kv[1])[0]
    most_likely_cause = sensor_prefixes[top_sensor]

    return most_likely_cause, feature_contributions
