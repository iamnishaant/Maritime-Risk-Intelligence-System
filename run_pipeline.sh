#!/bin/bash
set -e

# 🚢 Maritime Disruption Intelligence System - Automated Pipeline Runner
# This script executes the entire end-to-end AI pipeline and verifies each step.

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "--------------------------------------------------------"
echo "🚢 Initializing Maritime AI Pipeline Execution"
echo "--------------------------------------------------------"

# 1. PHASE 01: AIS INGESTION
echo -e "\n[1/5] Running Phase 01: AIS Data Ingestion..."
spark-submit --driver-memory 8g src/phase01_ingestion.py
if [ -d "data/processed/clean_ais.parquet" ]; then
    echo -e "${GREEN}✅ Phase 01 Complete: Clean AIS Parquet created.${NC}"
else
    echo -e "${RED}❌ Phase 01 Failed: Output not found.${NC}"
    exit 1
fi

# 2. WEATHER INGESTION
echo -e "\n[2/5] Running Weather Data Ingestion..."
spark-submit src/weather_ingestion.py
if [ -d "data/processed/weather.parquet" ]; then
    echo -e "${GREEN}✅ Weather Ingestion Complete: Weather Parquet created.${NC}"
else
    echo -e "${RED}❌ Weather Ingestion Failed: Output not found.${NC}"
    exit 1
fi

# 3. PHASE 02: FEATURE FUSION & MLlib ANOMALY
echo -e "\n[3/5] Running Phase 02: Feature Fusion & MLlib Anomaly Detection..."
spark-submit --driver-memory 8g src/phase02_feature_fusion.py
if [ -d "data/processed/fused_features.parquet" ] && [ -d "data/processed/dynamic_routes.parquet" ]; then
    echo -e "${GREEN}✅ Phase 02 Complete: Fused Features & Dynamic Routes created.${NC}"
else
    echo -e "${RED}❌ Phase 02 Failed: Features or Routes Parquet not found.${NC}"
    exit 1
fi

# 4. PHASE 03: GRAPH INTELLIGENCE
echo -e "\n[4/5] Running Phase 03: Graph Risk Propagation..."
spark-submit --driver-memory 4g src/phase03_pro_graph.py
if [ -f "data/processed/final_risk_scores.csv" ]; then
    echo -e "${GREEN}✅ Phase 03 Complete: Final Risk Scores CSV created.${NC}"
else
    echo -e "${RED}❌ Phase 03 Failed: Risk Scores CSV not found.${NC}"
    exit 1
fi

# 5. PHASE 04: FINAL EVALUATION & ML WEIGHTS
echo -e "\n[5/5] Running Phase 04: Final ML Evaluation & Report..."
spark-submit --driver-memory 4g src/phase04_final_model.py
if [ -f "data/results/final_research_report_upgraded.csv" ]; then
    echo -e "${GREEN}✅ Phase 04 Complete: Final Research Report generated.${NC}"
else
    echo -e "${RED}❌ Phase 04 Failed: Final Report not found.${NC}"
    exit 1
fi

echo -e "\n--------------------------------------------------------"
echo -e "${GREEN}🎉 ALL PIPELINE PHASES COMPLETED SUCCESSFULLY!${NC}"
echo "--------------------------------------------------------"
echo "Next Step: Launch the dashboard to visualize results:"
echo "👉 streamlit run src/app_dashboard.py"
echo "--------------------------------------------------------"
