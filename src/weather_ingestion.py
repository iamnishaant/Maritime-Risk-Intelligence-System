from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, date_trunc
import requests
import time
import pandas as pd

spark = SparkSession.builder.appName("Weather_Live_Ingestion").getOrCreate()

# Use the key from your screenshot
API_KEY = "13e933b4af2960b29ab7d5514794acfc"

# Define ports (Update these lats/lons to match your AIS data areas)
PORTS = [
    ("PORT_SINGAPORE", 1.5, 104.0),
    ("PORT_DUBAI", 25.0, 55.0),
    ("PORT_ROTTERDAM", 51.5, 4.0),
    ("PORT_SHANGHAI", 31.0, 121.0),
    ("PORT_MUMBAI", 19.0, 73.0),
    ("PORT_LOS_ANGELES", 34.0, -118.0),
    ("PORT_HAMBURG", 53.5, 10.0),
    ("PORT_TOKYO", 35.5, 140.0),
    ("PORT_SYDNEY", -33.5, 151.0),
    ("PORT_CAPE_TOWN", -33.5, 19.0)
]

records = []

print("Starting API Data Collection...")

for port_id, lat, lon in PORTS:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
        res = requests.get(url).json()

        # Robust data extraction
        if 'wind' in res:
            wind = res['wind'].get('speed', 0)
            rain = res.get('rain', {}).get('1h', 0)
            
            # Research Logic: Normalize weather risk between 0 and 1
            weather_score = min((wind/20) + (rain/10), 1.0)

            records.append({
                "port_id": port_id,
                "latitude": lat,
                "longitude": lon,
                "time": pd.Timestamp.now(),
                "weather_score": float(weather_score)
            })
            print(f"Fetched data for {port_id}: Score {weather_score}")
        else:
            print(f"⚠️ Warning: Unexpected response for {port_id}: {res.get('message', 'No wind data')}")

    except Exception as e:
        print(f"❌ Error fetching {port_id}: {e}")
    
    time.sleep(1) # Respect API rate limits

if records:
    # Convert to Spark
    pdf = pd.DataFrame(records)
    df = spark.createDataFrame(pdf)

    # Standardize time to the hour for joining in Phase 2
    df = df.withColumn("time", date_trunc("hour", to_timestamp("time")))

    # Save to Silver Layer
    df.write.mode("overwrite").parquet("data/processed/weather.parquet")
    print("✅ Live Weather Parquet created successfully!")
else:
    print("No records collected. Check API key propagation.")
