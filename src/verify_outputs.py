from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Verify_Data").getOrCreate()

print("\n--- 🌦️ WEATHER DATA CHECK ---")
try:
    weather_df = spark.read.parquet("../data/processed/weather.parquet")
    weather_df.show(5)
    print(f"Total Weather Records: {weather_df.count()}")
except Exception as e:
    print(f"Error reading weather: {e}")

print("\n--- 🚢 AIS DATA CHECK ---")
try:
    ais_df = spark.read.parquet("../data/processed/clean_ais.parquet")
    ais_df.show(5)
    print(f"Total AIS Records: {ais_df.count()}")
except Exception as e:
    print(f"Error reading AIS: {e}")
