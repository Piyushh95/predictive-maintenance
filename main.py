import pandas as pd
import matplotlib.pyplot as plt
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
    analyze_failure_cause,
    train_failure_model,
    predict_failure_probability,
)

# 1. Load data
df = pd.read_csv("data/sensor_data.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")

# 2. Feature engineering
feat_df = create_rolling_features(df)

X = feat_df.drop(columns=["timestamp"])

# 3. Train on early normal data (first 30%)
split = int(0.3 * len(X))

scaler = StandardScaler()
scaler.fit(X.iloc[:split])

X_scaled = scaler.transform(X)
X_train = X_scaled[:split]

model = train_model(X_train)

# 4. Anomaly scoring
scores = anomaly_score(model, X_scaled)
feat_df["anomaly_score"] = scores

# 5. Health score
min_s, max_s = scores.min(), scores.max()
feat_df["health"] = anomaly_to_health(scores, min_s, max_s)

# 5b. Failure probability model (RandomForest on health-derived labels)
health_scores_train = feat_df["health"].iloc[:split].values
failure_model = train_failure_model(X_train, health_scores_train)
latest_feature_vector = X_scaled[-1]
failure_prob = predict_failure_probability(failure_model, latest_feature_vector)
print(f"Failure Probability: {failure_prob * 100:.2f}%")

# 6. Degradation + alert escalation
signals = compute_degradation_signals(feat_df["health"])
feat_df = pd.concat([feat_df, signals], axis=1)
feat_df["alert_level"] = assign_alert_level(
    feat_df["health"],
    degradation_trend=feat_df["degradation_trend"],
    very_fast_degradation=feat_df["very_fast_degradation"],
)

# 7. Remaining Useful Life (RUL) prediction in hours + failure report
try:
    rul_info = predict_rul(feat_df)
    print(
        "Predicted failure time:",
        rul_info["predicted_failure_time"],
        "| Remaining hours:",
        f"{rul_info['remaining_hours']:.2f}",
    )
except Exception as exc:
    rul_info = None
    print(f"RUL prediction not available (time-based model): {exc}")

if feat_df["alert_level"].isin(["Critical", "Emergency"]).any() and rul_info is not None:
    generate_failure_report(df, feat_df, rul_info, output_path="machine_report.txt")
    print("Failure report written to machine_report.txt")
else:
    print("No Critical/Emergency alerts or time-based RUL unavailable; failure report not generated.")

# 8. Additional degradation detection and RUL in timesteps (portfolio-style)
degr_start_idx = detect_degradation_start(feat_df["health"].values)
if degr_start_idx is not None:
    print(f"Degradation start detected (rolling slope) at index: {degr_start_idx}")
else:
    print("Degradation start (rolling slope) not detected.")

rul_steps = estimate_rul(feat_df["health"].values, threshold=30)
if rul_steps is not None:
    print(f"Estimated RUL: {rul_steps} timesteps until health reaches 30.")
else:
    print("RUL in timesteps could not be estimated (no clear degrading trend).")

# 9. Current machine status and RandomForest-based root cause analysis
current_health = float(feat_df["health"].iloc[-1])
machine_status = get_alert_level(current_health)
print(f"Machine Status (latest point): {machine_status}")

# Consolidated results block for quick operational readout.
print("\nExample Output:")
print(f"- Failure Probability: {failure_prob * 100:.2f}%")
if rul_info is not None:
    print(f"- RUL: {rul_info['remaining_hours']:.2f} hours")
else:
    print("- RUL: not available")
print(f"- Status: {str(machine_status).upper()}")

# Maintenance recommendation combining probability + alert severity.
latest_alert_level = str(feat_df["alert_level"].iloc[-1])
if (
    failure_prob > 0.6
    and latest_alert_level.upper() in ("CRITICAL", "EMERGENCY")
):
    print("Maintenance required within next operational cycle")

if machine_status in ("CRITICAL", "EMERGENCY"):
    # Build a binary label: failure vs normal, based on time-series alert levels.
    failure_mask = feat_df["alert_level"].isin(["Critical", "Emergency", "CRITICAL", "EMERGENCY"])
    y_rca = failure_mask.astype(int)

    # Use all vibration/temperature/pressure-related features as inputs.
    sensor_feature_cols = [
        c
        for c in feat_df.columns
        if any(c.startswith(prefix) for prefix in ("vibration", "temperature", "pressure"))
    ]

    if sensor_feature_cols:
        X_rca = feat_df[sensor_feature_cols]
        feature_importance = root_cause_analysis(X_rca, y_rca)

        # Convert feature importances to percentages and pick top contributors.
        total_imp = sum(feature_importance.values())
        if total_imp > 0:
            contributions_pct = {
                name: (imp / total_imp) * 100.0 for name, imp in feature_importance.items()
            }
        else:
            contributions_pct = {name: 0.0 for name in feature_importance.keys()}

        # For a concise explanation, aggregate by sensor family.
        sensor_families = {
            "vibration": "bearing wear",
            "temperature": "lubrication or cooling issue",
            "pressure": "valve or blockage",
        }
        family_scores = {k: 0.0 for k in sensor_families.keys()}
        for feat, pct in contributions_pct.items():
            for fam in family_scores.keys():
                if feat.startswith(fam):
                    family_scores[fam] += pct
                    break

        top_family, top_pct = max(family_scores.items(), key=lambda kv: kv[1])
        cause_text = sensor_families[top_family]
        print(
            f"Failure likely due to {cause_text} "
            f"({top_family} contributing {top_pct:.1f}% of model importance)."
        )
        print("Top contributing sensor features:")
        for name, pct in sorted(contributions_pct.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            print(f"  - {name}: {pct:.1f}%")
    else:
        print("Root cause analysis skipped: no sensor feature columns available.")
else:
    print("Root cause analysis skipped: machine is not in CRITICAL or EMERGENCY state.")

def plot_health(feat_df):
    """
    Interactive helper to visualize machine health and alert levels.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(
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
        plt.scatter(
            feat_df.loc[mask, "timestamp"],
            feat_df.loc[mask, "health"],
            color=color,
            s=12 if level != "Normal" else 8,
            alpha=0.9 if level != "Normal" else 0.6,
            label=level,
        )

    plt.axhline(70, color="yellow", linestyle="--", linewidth=1, label="Warning Threshold (70)")
    plt.axhline(50, color="orange", linestyle="--", linewidth=1, label="Critical Threshold (50)")
    plt.axhline(30, color="red", linestyle="--", linewidth=1, label="Emergency Threshold (30)")
    plt.legend()
    plt.title("Machine Health Over Time")
    plt.show()


def interactive_console(
    df,
    feat_df,
    rul_info,
    failure_prob,
    machine_status,
    degr_start_idx,
    rul_steps,
):
    """
    Simple text-based interface to explore model outputs interactively.
    """
    while True:
        print("\n=== Predictive Maintenance Console ===")
        print("1) Show health and alerts plot")
        print("2) Show latest KPIs and status")
        print("3) Generate / refresh failure report")
        print("4) Exit")
        choice = input("Select an option [1-4]: ").strip()

        if choice == "1":
            plot_health(feat_df)
        elif choice == "2":
            print("\n--- Latest KPIs ---")
            print(f"Latest health score: {feat_df['health'].iloc[-1]:.2f}")
            
            print(f"Machine status: {machine_status}")
            print(f"Failure probability: {failure_prob * 100:.2f}%")
            if rul_info is not None:
                print(f"Time-based RUL (hours): {rul_info['remaining_hours']:.2f}")
            if degr_start_idx is not None:
                print(f"Degradation start index (rolling slope): {degr_start_idx}")
            if rul_steps is not None:
                print(f"Estimated RUL (timesteps): {rul_steps}")
        elif choice == "3":
            if rul_info is None:
                print("Cannot generate failure report: time-based RUL is not available.")
            else:
                generate_failure_report(df, feat_df, rul_info, output_path="machine_report.txt")
                print("Failure report written to machine_report.txt")
        elif choice == "4":
            print("Exiting console.")
            break
        else:
            print("Invalid choice, please select 1-4.")


# Launch interactive console at the end of the analysis run.
interactive_console(
    df=df,
    feat_df=feat_df,
    rul_info=rul_info,
    failure_prob=failure_prob,
    machine_status=machine_status,
    degr_start_idx=degr_start_idx,
    rul_steps=rul_steps,
)
