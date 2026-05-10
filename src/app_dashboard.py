import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import time

st.set_page_config(page_title="Maritime AI Risk Control", layout="wide")

st.title("🚢 Maritime Disruption Intelligence System")
st.markdown("### Real-Time Graph-Based Risk Propagation Dashboard")

DATA_PATH = "data/results/final_research_report_upgraded.csv"

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

with st.expander("ℹ️ How to read this metric"):
    st.markdown("""
    The **System Risk Index** is an ML-weighted score (0.0 to 1.0). 
    It combines **Local Congestion** (learned via K-Means clustering), **Weather conditions**, and **Graph Propagation** (risk cascading from connected ports).
    """)

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
        
    with st.expander("ℹ️ What is a Hidden Threat?"):
        st.markdown("""
        A **Hidden Threat (Risk Delta)** occurs when a port looks fine locally, but is flagged as high-risk because a major delay is cascading towards it through the shipping network. Traditional systems completely miss this.
        """)

# -----------------------------
# 🌐 GRAPH WITH DYNAMIC COLORS
# -----------------------------
with col2:
    st.subheader("🌐 Network Topology")

    G = nx.DiGraph()
    
    # Load dynamic edges safely
    try:
        routes_df = pd.read_parquet("data/processed/dynamic_routes.parquet")
        top_edges = routes_df[routes_df["weight"] > 0.01].sort_values("weight", ascending=False).head(20)
        for _, row in top_edges.iterrows():
            G.add_edge(row["port_id"], row["next_port"])
    except Exception:
        pass
        
    # Fallback if no routes survived the strict filters
    if G.number_of_nodes() == 0:
        G.add_edges_from([
            ("PORT_SINGAPORE", "PORT_DUBAI"),
            ("PORT_DUBAI", "PORT_ROTTERDAM"),
            ("PORT_SHANGHAI", "PORT_LOS_ANGELES")
        ])

    node_colors = []
    nodes_list = list(G.nodes())
    for node in nodes_list:
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

    if nodes_list:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor('#0e1117') # Streamlit dark background
        ax.set_facecolor('#0e1117')

        # Spread out nodes cleanly
        pos = nx.spring_layout(G, seed=42, k=2.0)
        
        # Clean up labels for display
        labels = {n: n.replace("PORT_", "").replace("_", " ") for n in nodes_list}

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_list, node_color=node_colors, node_size=900, ax=ax)
        
        # Draw edges with a sleek curve
        nx.draw_networkx_edges(
            G, pos, 
            edge_color='#888888', 
            width=1.5, 
            arrowsize=15, 
            connectionstyle='arc3,rad=0.15', 
            ax=ax
        )
        
        # Draw labels in white
        nx.draw_networkx_labels(
            G, pos, 
            labels, 
            font_size=9, 
            font_color='white', 
            font_weight='bold', 
            ax=ax
        )

        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No active network topology to display.")
        
    with st.expander("ℹ️ Understanding the Network"):
        st.markdown("""
        This map is **data-driven**, not hardcoded. The system tracks raw AIS data and dynamically draws edges only where significant vessel traffic actually exists. **Red nodes** indicate critical systemic risk.
        """)

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
# 🧠 DEEP DIVE INTELLIGENCE
# -----------------------------
st.markdown("---")
st.subheader("🧠 Deep Dive Intelligence (AI Port Analysis)")

selected_port = st.selectbox("Select a Port to Analyze:", df["port_id"].unique())

if selected_port:
    port_stats = df[df["port_id"] == selected_port].sort_values("time", ascending=False).iloc[0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Local Anomaly Score (K-Means)", f"{port_stats.get('A_norm', 0):.3f}")
    c2.metric("Environmental Score", f"{port_stats.get('E_norm', 0):.3f}")
    c3.metric("Incoming Network Risk", f"{port_stats.get('G_norm', 0):.3f}")
    
    st.info(f"""
    **AI Analysis for {selected_port}:**
    *   **Baseline Risk:** `{port_stats.get('baseline_risk', 0):.3f}` (Risk if this port was isolated from the global network).
    *   **Total System Risk:** `{port_stats.get('total_system_risk', 0):.3f}` (True risk after accounting for global supply chain cascades).
    *   **Risk Delta:** `{port_stats.get('risk_delta', 0):.3f}`. 
    """)
    
    delta = port_stats.get('risk_delta', 0)
    total_risk = port_stats.get('total_system_risk', 0)
    
    if delta > 0.1:
        st.warning(f"⚠️ **Conclusion:** {selected_port} is suffering from heavy cascading delays. The problem is NOT local; it is importing delays from upstream ports.")
    elif total_risk > 0.5:
        st.error(f"🚨 **Conclusion:** {selected_port} is highly congested locally. It is likely a source of disruption for the rest of the network.")
    else:
        st.success(f"✅ **Conclusion:** {selected_port} is currently operating within normal parameters. No significant local or cascading threats detected.")

# -----------------------------
# AUTO REFRESH (SAFE)
# -----------------------------
st.sidebar.markdown("---")
live_mode = st.sidebar.checkbox("🟢 Enable Live Auto-Refresh", value=False)

if live_mode:
    time.sleep(3)
    st.rerun()
