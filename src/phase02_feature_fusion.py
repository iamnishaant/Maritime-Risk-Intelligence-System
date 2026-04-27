from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# 1. START SPARK
# -----------------------------
spark = SparkSession.builder \
    .appName("Phase2_Feature_Fusion_Final") \
    .getOrCreate()

print("🚀 Spark Started")

# -----------------------------
# 2. LOAD DATA
# -----------------------------
ais_df = spark.read.parquet("../data/processed/clean_ais.parquet")
weather_df = spark.read.parquet("../data/processed/weather.parquet")

print("✅ Data Loaded")

# -----------------------------
# FIX SWAPPED COORDINATES
# -----------------------------
ais_df = ais_df.withColumnRenamed("latitude", "temp") \
               .withColumnRenamed("longitude", "latitude") \
               .withColumnRenamed("temp", "longitude")

# -----------------------------
# 3. PORT MAPPING (SPATIAL)
# -----------------------------
ais_df = ais_df.withColumn(
    "port_id",
    F.when(F.col("latitude").between(50, 53), "PORT_A")
     .when(F.col("latitude").between(39, 42), "PORT_B")
     .when(F.col("latitude").between(33, 35), "PORT_C")
     .otherwise("SEA")
)

# -----------------------------
# 4. FILTER ONLY PORT DATA
# -----------------------------
ais_df = ais_df.filter(F.col("port_id") != "SEA")

print("✅ Port Mapping Done")

# -----------------------------
# -----------------------------
# 5. SPATIO-TEMPORAL JOIN (FIXED)
# -----------------------------
fused_df = ais_df.join(
    weather_df,
    (ais_df.port_id == weather_df.port_id) &
    (ais_df.time == weather_df.time),
    "left"
).select(
    ais_df.port_id,
    ais_df.vessel_id,
    ais_df.timestamp,
    ais_df.time,
    ais_df.speed,
    weather_df.weather_score
)
# Fill missing weather values
fused_df = fused_df.fillna({"weather_score": 0})

print("✅ Join Completed")

# -----------------------------
# 6. DWELL TIME CALCULATION
# -----------------------------
window_spec = Window.partitionBy("vessel_id").orderBy("timestamp")

fused_df = fused_df.withColumn(
    "time_diff",
    F.unix_timestamp("timestamp") -
    F.lag(F.unix_timestamp("timestamp")).over(window_spec)
)

# Convert to hours
fused_df = fused_df.withColumn(
    "time_diff_hours",
    F.col("time_diff") / 3600
)

# Stationary condition
fused_df = fused_df.withColumn(
    "is_stationary",
    F.when(F.col("speed") < 1, 1).otherwise(0)
)

# Dwell time
fused_df = fused_df.withColumn(
    "dwell_time",
    F.when(F.col("is_stationary") == 1, F.col("time_diff_hours")).otherwise(0)
)

print("✅ Dwell Time Computed")

# -----------------------------
# 7. AGGREGATION (PORT LEVEL)
# -----------------------------
agg_df = fused_df.groupBy("port_id", "time").agg(
    F.avg("dwell_time").alias("avg_dwell"),
    F.count("vessel_id").alias("vessel_count"),
    F.avg("weather_score").alias("weather_score")
)

print("✅ Aggregation Done")

# -----------------------------
# 8. PROXY ANOMALY (Z-SCORE)
# -----------------------------
stats = agg_df.select(
    F.mean("avg_dwell").alias("mean_dwell"),
    F.stddev("avg_dwell").alias("std_dwell")
).collect()[0]

mean_dwell = stats["mean_dwell"]
std_dwell = stats["std_dwell"]

agg_df = agg_df.withColumn(
    "anomaly_score",
    (F.col("avg_dwell") - mean_dwell) / std_dwell
)

print("✅ Anomaly Score Computed")

# -----------------------------
# 9. SAVE OUTPUT
# -----------------------------
output_path = "../data/processed/fused_features.parquet"

agg_df.write.mode("overwrite").parquet(output_path)

print("🎉 Phase 2 COMPLETE — Features Ready!")

# -----------------------------
# 10. VERIFY
# -----------------------------
spark.read.parquet(output_path).show(5)
