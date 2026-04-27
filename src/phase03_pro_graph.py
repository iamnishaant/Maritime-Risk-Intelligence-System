import networkx as nx
import pandas as pd
from pyspark.sql import SparkSession

# -----------------------------
# 1. LOAD DATA
# -----------------------------
spark = SparkSession.builder.appName("Phase3_Pro_Graph").getOrCreate()

df = spark.read.parquet("../data/processed/fused_features.parquet")
pdf = df.toPandas()

# -----------------------------
# 2. GET LATEST STATE
# -----------------------------
latest = pdf.sort_values("time").groupby("port_id").tail(1).copy()

# Normalize vessel count
max_vessels = latest["vessel_count"].max()
latest["normalized_load"] = latest["vessel_count"] / max_vessels

# -----------------------------
# 3. BUILD GRAPH (DIRECTED)
# -----------------------------
G = nx.DiGraph()

ports = latest["port_id"].tolist()

# Add ALL nodes first (IMPORTANT)
for p in ports:
    G.add_node(p)

# Define routes (can be improved later)
routes = [
    ("PORT_C", "PORT_B"),
    ("PORT_B", "PORT_A"),
    ("PORT_C", "PORT_A")
]

# Add edges with dynamic weights
for u, v in routes:
    if u in ports and v in ports:
        source_load = latest[latest["port_id"] == u]["normalized_load"].values[0]
        G.add_edge(u, v, weight=float(source_load))

# -----------------------------
# 4. INITIAL RISK
# -----------------------------
risk_dict = dict(zip(latest["port_id"], latest["anomaly_score"]))
nx.set_node_attributes(G, risk_dict, "risk")

# -----------------------------
# 5. MULTI-STEP PROPAGATION
# -----------------------------
gamma = 0.75
iterations = 3   # multi-step propagation

for _ in range(iterations):
    new_risk = {}
    
    for node in G.nodes():
        current = G.nodes[node]["risk"]
        
        # incoming influence
        influence = 0
        for upstream in G.predecessors(node):
            weight = G[upstream][node]["weight"]
            upstream_risk = G.nodes[upstream]["risk"]
            influence += weight * upstream_risk
        
        new_risk[node] = current + gamma * influence
    
    # update
    for node in G.nodes():
        G.nodes[node]["risk"] = new_risk[node]

# -----------------------------
# 6. SAVE FINAL RESULTS
# -----------------------------
results = []

for node, data in G.nodes(data=True):
    results.append({
        "port_id": node,
        "final_risk": data["risk"]
    })

result_df = pd.DataFrame(results)
result_df.to_csv("../data/processed/final_risk_scores.csv", index=False)

# -----------------------------
# 7. DISPLAY
# -----------------------------
print(f"{'Port':<10} | {'Final Risk':<15}")
print("-" * 30)

for node, data in G.nodes(data=True):
    print(f"{node:<10} | {data['risk']:<15.4f}")
