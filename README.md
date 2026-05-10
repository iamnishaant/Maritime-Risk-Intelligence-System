# Maritime Disruption Intelligence System

## Overview
Global maritime logistics is highly interconnected. A local disruption at a single port—due to congestion, severe weather, or operational delays—can rapidly propagate across the shipping network, resulting in cascading supply chain failures. The **Maritime Disruption Intelligence System** is a research-grade, big-data platform that detects early disruption signals and dynamically models how risk spreads across interconnected port networks using AIS (Automatic Identification System) tracking, live weather signals, and temporal graph-based risk propagation.

## Problem Statement
In the domain of Maritime Logistics and Supply Chain Management, proactively identifying and mitigating disruptions is a critical challenge. Modern shipping relies on tightly-coupled schedules. When one node (port) fails or slows down, the entire network suffers. 

Existing maritime monitoring systems primarily focus on localized, reactive anomaly detection. They fail to capture the **network dependencies** between ports and are unable to predict the ripple effects of a disruption. 

## Motivation
This project was built to address the lack of proactive, network-aware risk modeling in global shipping. In real-world scenarios, a storm hitting a port in Asia will inevitably cause cascading delays in European or American ports weeks later. By identifying these "hidden risks" (risks induced by the network rather than localized events) early on, port authorities and logistics providers can reroute vessels, optimize supply chains, and save millions in demurrage and delay costs.

## Related Work / Existing Solutions
Traditional approaches to maritime risk assessment include:
* **Standalone Anomaly Detection**: Models that track vessel speeds or port congestion in isolation. 
* **Static Risk Dashboards**: Basic BI tools showing current delays without predictive capabilities.

**Limitations of Existing Approaches:**
* **Lack of Network Context**: They ignore how ports are structurally connected, failing to anticipate cascading delays.
* **High Noise & False Positives**: Reactive models often flag standard operational fluctuations as anomalies.
* **Late Detection**: By the time a delay is registered at a localized level, the systemic disruption has already occurred, offering no actionable lead time.

## Proposed Approach
We propose a **Temporal Graph-Based Risk Model** that fuses localized anomalies, external weather factors, and network topology into a unified risk metric. 

1. **Multi-Source Data Fusion**: Ingests high-throughput AIS vessel coordinates and live OpenWeatherMap API data using Apache Spark.
2. **Proxy Anomaly Detection**: Computes localized congestion anomalies by analyzing vessel dwell times and calculating Z-scores.
3. **Graph Construction**: Models ports as nodes and shipping routes as directed edges. Edge weights are dynamically calculated based on normalized vessel loads (traffic volume).
4. **Multi-Step Risk Propagation**: Uses a spatial decay factor ($\gamma$) to propagate risk iteratively from upstream ports to downstream ports.
5. **Temporal Smoothing**: Applies Exponentially Weighted Moving Averages (EWMA) to risk signals to reduce noise and prevent transient false alarms.

## System Architecture

The pipeline is entirely modular and scalable, built on Apache Spark for distributed processing.

* **Data Ingestion (`phase01_ingestion.py`, `weather_ingestion.py`)**: Cleans and standardizes raw AIS data and live weather data into an optimized Parquet format.
* **Feature Fusion (`phase02_feature_fusion.py`)**: Performs spatio-temporal joins. Maps coordinates to specific ports, calculates stationary dwell times, and generates baseline anomaly scores.
* **Graph Propagation Engine (`phase03_pro_graph.py`)**: Utilizes `NetworkX` to build a directed graph. Injects initial anomalies and propagates risk multi-step using historical routing logic.
* **Final Evaluation Model (`phase04_final_model.py`)**: Computes the Total System Risk, baseline comparisons, Risk Delta, and Mean Time To Detection (MTTD).
* **Streaming & Visualization (`phase05_streaming.py`, `app_dashboard.py`)**: Provides a real-time CLI stream simulation and an interactive Streamlit dashboard mapping dynamic network topologies.

## Key Features
* 🔥 **Hybrid Risk Equation**: Computes total risk as $R_i(t) = \alpha A_i(t) + \beta G_i(t) + \gamma E_i(t)$ where $A$ is local anomaly, $G$ is graph influence, and $E$ is environmental factors.
* 🌐 **Dynamic Network Topology**: Edges between ports react to live vessel loads, adjusting the severity of risk propagation.
* ⏱️ **Temporal Risk Smoothing**: Implements EWMA to stabilize volatile risk scores.
* ⚠️ **Hidden Threat Detection (Risk Delta)**: Specifically isolates risks that are purely network-induced, which standard models would miss entirely.

## Results & Metrics
The effectiveness of the system is quantified using specific analytical metrics:
* **Mean Time To Detection (MTTD)**: Measures the time differential (in hours) between when a cascading risk originates and when the system successfully flags a high-risk threshold.
* **Risk Delta**: Evaluates $Total\_System\_Risk - Baseline\_Risk$. A high Delta indicates the system successfully discovered a hidden, propagated risk that a non-graph baseline model missed.
* **Ablation Studies**: Empirically proves that the inclusion of the Graph component ($\beta G_i(t)$) significantly elevates the predictive accuracy of the model over standard anomaly detection.

## Installation & Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd project

# 2. Install dependencies
pip install -r requirements.txt
# Ensure Apache Spark / PySpark is configured on your system.
```

## Usage

**1. Data Ingestion & Preprocessing**
```bash
spark-submit src/phase01_ingestion.py
python3 src/weather_ingestion.py
```

**2. Feature Fusion & Graph Processing**
```bash
spark-submit src/phase02_feature_fusion.py
spark-submit src/phase03_pro_graph.py
```

**3. Model Evaluation**
```bash
spark-submit src/phase04_final_model.py
```

**4. Real-Time Interfaces**
```bash
# Run CLI Streaming Simulation
python3 src/phase05_streaming.py

# Launch Interactive Web Dashboard
streamlit run src/app_dashboard.py
```

## Project Structure

```
project/
├── data/
│   ├── processed/          # Intermediate Parquet files
│   ├── raw/                # Raw AIS CSV data
│   └── results/            # Final evaluation metrics & reports
├── src/
│   ├── app_dashboard.py           # Streamlit UI
│   ├── phase01_ingestion.py       # Spark AIS Ingestion
│   ├── phase02_feature_fusion.py  # Spatio-temporal join & Anomaly scoring
│   ├── phase03_pro_graph.py       # NetworkX risk propagation
│   ├── phase04_final_model.py     # Total risk computation & MTTD
│   ├── phase05_streaming.py       # Live stream simulation
│   ├── verify_outputs.py          # Sanity check scripts
│   └── weather_ingestion.py       # OpenWeatherMap API integration
└── README.md
```

## Limitations
* **Static Routing**: Current graph edges (routes between ports) are hardcoded for `PORT_C -> PORT_B -> PORT_A`.
* **Basic Environmental Model**: The weather scoring is normalized using simple wind and rain heuristics rather than advanced meteorological modeling.
* **Graph Scale**: Operations currently run in-memory via NetworkX, which may bottleneck at a massive global scale (thousands of ports and millions of vessels).

## Future Work
* **Dynamic Edge Extraction**: Algorithmically infer shipping routes and edge weights directly from historical AIS trajectories.
* **Graph Neural Networks (GNNs)**: Upgrade from multi-step propagation to Graph Convolutional Networks (GCN) or Temporal Graph Networks (TGN) for learning complex, non-linear risk representations.
* **Apache Kafka Integration**: Transition the batch-oriented Spark jobs to continuous `Spark Structured Streaming` paired with Kafka topics.
* **Spark GraphX**: Migrate from NetworkX to Spark GraphX or GraphFrames to handle extreme-scale graph computations efficiently.

## Contributing
Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change. 
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed for academic and research purposes.
