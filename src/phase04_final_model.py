import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Phase4_Final_Evaluation").getOrCreate()

# -----------------------------
# 1. LOAD DATA
# -----------------------------
features_df = spark.read.parquet("../data/processed/fused_features.parquet").toPandas()
graph_df = pd.read_csv("../data/processed/final_risk_scores.csv")

# Rename for clarity
graph_df = graph_df.rename(columns={"final_risk": "graph_risk"})

# -----------------------------
# 2. MERGE
# -----------------------------
df = features_df.merge(graph_df, on="port_id", how="left")

# -----------------------------
# 3. NORMALIZATION (CRITICAL)
# -----------------------------
def normalize(col):
    return (col - col.min()) / (col.max() - col.min() + 1e-6)

df["A_norm"] = normalize(df["anomaly_score"])
df["G_norm"] = normalize(df["graph_risk"])
df["E_norm"] = normalize(df["weather_score"])

# -----------------------------
# 4. RISK COMPUTATION
# -----------------------------
alpha, beta, gamma = 0.5, 0.3, 0.2

df["baseline_risk"] = (alpha * df["A_norm"]) + (gamma * df["E_norm"])

df["total_system_risk"] = (
    alpha * df["A_norm"] +
    beta * df["G_norm"] +
    gamma * df["E_norm"]
)

# -----------------------------
# 5. RISK DELTA (KEY METRIC)
# -----------------------------
df["risk_delta"] = df["total_system_risk"] - df["baseline_risk"]

# -----------------------------
# 5.5 MTTD (Mean Time To Detection)
# -----------------------------

# Ensure time is datetime
df["time"] = pd.to_datetime(df["time"])

# Define high-risk threshold (top 10%)
threshold = df["total_system_risk"].quantile(0.9)

# Create detection flag
df["high_risk_flag"] = df["total_system_risk"] > threshold

# -----------------------------
# BASIC MTTD (Timestamp)
# -----------------------------
mttd_time = df[df["high_risk_flag"]].groupby("port_id")["time"].min()

print("\n=== MTTD (First Detection Time) ===")
print(mttd_time)

# -----------------------------
# ADVANCED MTTD (Time Difference in Hours)
# -----------------------------

# Compute time difference from earliest timestamp per port
mttd_hours = df[df["high_risk_flag"]].groupby("port_id").apply(
    lambda x: (x["time"].min() - df[df["port_id"] == x.name]["time"].min()).total_seconds() / 3600
)

print("\n=== MTTD (Hours to Detection) ===")
print(mttd_hours)



# -----------------------------
# 6. DATA-DRIVEN THRESHOLDS
# -----------------------------
q1 = df["total_system_risk"].quantile(0.33)
q2 = df["total_system_risk"].quantile(0.66)

def get_status(r):
    if r > q2: return "CRITICAL"
    if r > q1: return "ELEVATED"
    return "STABLE"

df["final_status"] = df["total_system_risk"].apply(get_status)

# -----------------------------
# 🆕 ADDED: TEMPORAL RISK MODEL (NEW)
# -----------------------------

lambda_ = 0.7

df = df.sort_values(["port_id", "time"])

df["temporal_risk"] = df.groupby("port_id")["total_system_risk"].transform(
    lambda x: x.ewm(alpha=(1-lambda_)).mean()
)
# -----------------------------
# 7. SAVE OUTPUT
# -----------------------------
df.to_csv("../data/results/final_research_report_upgraded.csv", index=False)

# -----------------------------
# 8. SUMMARY
# -----------------------------
summary = df.groupby("port_id")[["baseline_risk", "total_system_risk", "risk_delta"]].mean()

print("\n=== RESEARCH SUMMARY ===")
print(summary)

# -----------------------------
# 9. VISUALIZATION
# -----------------------------
summary[["baseline_risk", "total_system_risk"]].plot(kind="bar")
plt.title("Baseline vs Proposed Model")
plt.ylabel("Normalized Risk")
plt.savefig("../data/results/risk_comparison_plot_upgraded.png")

print("✅ Phase 4 Complete")
