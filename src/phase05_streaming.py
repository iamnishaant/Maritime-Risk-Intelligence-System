import pandas as pd
import time
from datetime import datetime

# -----------------------------
# 1. LOAD DATA
# -----------------------------
try:
    df = pd.read_csv("../data/results/final_research_report_upgraded.csv")
except:
    print("❌ ERROR: File not found. Check path.")
    exit()

# Ensure time column is datetime
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Drop bad rows
df = df.dropna(subset=["time"])

# -----------------------------
# 2. STREAMING SIMULATION
# -----------------------------
print("📡 INITIALIZING MARITIME STREAMING INTERFACE...")
print("Connected to Virtual Topic: 'ais_realtime_risk'\n")

print(f"{'TIME':<10} | {'PORT':<10} | {'RISK':<10} | {'STATUS':<12} | {'THREAT'}")
print("-" * 70)

try:
    while True:
        # Sort by time (important for realism)
        for _, row in df.sort_values("time").iterrows():
            current_time = datetime.now().strftime("%H:%M:%S")

            port = row.get("port_id", "UNKNOWN")
            risk = row.get("total_system_risk", 0.0)
            status = row.get("final_status", "N/A")
            delta = row.get("risk_delta", 0.0)

            # Threat detection logic (your novelty)
            threat = "⚠️ CASCADING RISK" if delta > 0.2 else "NORMAL"

            print(f"{current_time:<10} | {port:<10} | {risk:<10.4f} | {status:<12} | {threat}")

            time.sleep(1.5)

except KeyboardInterrupt:
    print("\n🛑 Streaming stopped by user.")
