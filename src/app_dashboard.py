import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import time

st.set_page_config(page_title="Maritime AI Risk Control", layout="wide")

st.title("🚢 Maritime Disruption Intelligence System")
st.markdown("### Real-Time Graph-Based Risk Propagation Dashboard")

DATA_PATH = "../data/results/final_research_report_upgraded.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists(DATA_PATH):
    st.error("❌ Data file not found")
    st.stop()

df = pd.read_csv(DATA_PATH)

if df.empty:
    st.warning("⚠ No data available")
    st.stop()

df["time"] = pd.to_datetime(df["time"], errors="coerce")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("System Metrics")
st.sidebar.metric("Processing Engine", "Apache Spark")
st.sidebar.metric("Graph Engine", "NetworkX")
st.sidebar.metric("Data Points", len(df))

# -----------------------------
# 🔥 TOP RISK METRIC
# -----------------------------
top_port_row = df.sort_values("total_system_risk", ascending=False).iloc[0]

m1, m2, m3 = st.columns(3)
m1.metric("🔥 Highest Risk Node", top_port_row["port_id"])
m2.metric("📊 System Risk Index", f"{top_port_row['total_system_risk']:.3f}")
m3.metric("📡 Engine Status", "Active")

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# ⚠ HIDDEN RISK DETECTION
# -----------------------------
with col1:
    st.subheader("⚠️ Network-Induced Risks")

    hidden_df = df[df["risk_delta"] > 0.2][
        ['port_id', 'total_system_risk', 'risk_delta']
    ].drop_duplicates()

    if not hidden_df.empty:
        st.warning("Graph propagation detected hidden threats:")
        st.dataframe(hidden_df, width="stretch")
    else:
        st.success("No cascading risks detected.")

# -----------------------------
# 🌐 GRAPH WITH DYNAMIC COLORS
# -----------------------------
with col2:
    st.subheader("🌐 Network Topology")

    G = nx.DiGraph()
    G.add_edges_from([("PORT_C", "PORT_B"), ("PORT_B", "PORT_A")])

    node_colors = []
    for node in G.nodes():
        node_data = df[df["port_id"] == node]

        if not node_data.empty:
            node_risk = node_data["total_system_risk"].max()
        else:
            node_risk = 0

        if node_risk > 0.5:
            node_colors.append("#ff4b4b")  # Red
        elif node_risk > 0.3:
            node_colors.append("#ffa500")  # Orange
        else:
            node_colors.append("#2ecc71")  # Green

    fig, ax = plt.subplots(figsize=(5, 3.5))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2500,
        font_weight='bold',
        arrows=True,
        ax=ax
    )

    st.pyplot(fig)

# -----------------------------
# 📈 TEMPORAL ANALYSIS (SMOOTHED)
# -----------------------------
st.subheader("📈 Risk Trend (Raw vs Smoothed)")

chart_df = df.sort_values("time").set_index("time")

if "temporal_risk" in chart_df.columns:
    chart_df["smoothed_signal"] = chart_df["temporal_risk"].rolling(
        window=5, min_periods=1
    ).mean()

    st.line_chart(chart_df[["temporal_risk", "smoothed_signal"]])
    st.caption("Smoothed signal removes noise for stable decision-making.")
else:
    st.warning("Temporal risk not found")


# -----------------------------
# AUTO REFRESH (SAFE)
# -----------------------------
time.sleep(3)
st.rerun()
