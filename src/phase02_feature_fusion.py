from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# 1. START SPARK
# -----------------------------
spark = SparkSession.builder \
    .appName("Phase2_Feature_Fusion_Final") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

print("🚀 Spark Started")

# -----------------------------
# 2. LOAD DATA
# -----------------------------
ais_df = spark.read.parquet("data/processed/clean_ais.parquet")
weather_df = spark.read.parquet("data/processed/weather.parquet")

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
    F.when((F.col("latitude").between(1, 2)) & (F.col("longitude").between(103, 105)), "PORT_SINGAPORE")
     .when((F.col("latitude").between(24, 26)) & (F.col("longitude").between(54, 56)), "PORT_DUBAI")
     .when((F.col("latitude").between(51, 52)) & (F.col("longitude").between(3, 5)), "PORT_ROTTERDAM")
     .when((F.col("latitude").between(30, 32)) & (F.col("longitude").between(120, 122)), "PORT_SHANGHAI")
     .when((F.col("latitude").between(18, 20)) & (F.col("longitude").between(72, 74)), "PORT_MUMBAI")
     .when((F.col("latitude").between(33, 35)) & (F.col("longitude").between(-119, -117)), "PORT_LOS_ANGELES")
     .when((F.col("latitude").between(53, 54)) & (F.col("longitude").between(9, 11)), "PORT_HAMBURG")
     .when((F.col("latitude").between(35, 36)) & (F.col("longitude").between(139, 141)), "PORT_TOKYO")
     .when((F.col("latitude").between(-34, -33)) & (F.col("longitude").between(150, 152)), "PORT_SYDNEY")
     .when((F.col("latitude").between(-34, -33)) & (F.col("longitude").between(18, 20)), "PORT_CAPE_TOWN")
     .otherwise("SEA")
)

print("✅ Port Mapping Done")

# -----------------------------
# 3.5 DYNAMIC ROUTE LEARNING
# -----------------------------
# OPTIMIZATION: Repartition by vessel_id to avoid OOM during large window sorts
ais_df = ais_df.repartition("vessel_id")

route_window = Window.partitionBy("vessel_id").orderBy("time")

routes_df = ais_df.withColumn("next_port", F.lead("port_id").over(route_window)) \
                  .withColumn("next_time", F.lead("time").over(route_window))

routes_df = routes_df.withColumn(
    "time_diff_sec",
    F.unix_timestamp("next_time") - F.unix_timestamp("time")
)

routes_df = routes_df.filter(
    (F.col("port_id") != F.col("next_port")) &
    (F.col("port_id") != "SEA") &
    (F.col("next_port") != "SEA") &
    (F.col("next_port").isNotNull()) &
    (F.col("time_diff_sec") < 432000) # Relaxed to 5 days
)

routes_agg = routes_df.groupBy("port_id", "next_port").count()

# Noise filter (Relaxed for 5-day dataset)
routes_agg = routes_agg.filter(F.col("count") >= 1)

# Normalize per source port
port_window = Window.partitionBy("port_id")
routes_agg = routes_agg.withColumn(
    "weight",
    F.when(
        F.sum("count").over(port_window) > 0,
        F.col("count") / F.sum("count").over(port_window)
    ).otherwise(0)
)

try:
    print(f"Total dynamic routes learned: {routes_agg.count()}")
    routes_agg.write.mode("overwrite").parquet("data/processed/dynamic_routes.parquet")
    print("✅ Dynamic Routes Saved")
except Exception as e:
    print(f"⚠️ Warning: Could not save dynamic routes. Make sure data exists. Error: {e}")

# -----------------------------
# 4. FILTER ONLY PORT DATA
# -----------------------------
ais_df = ais_df.filter(F.col("port_id") != "SEA")

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
# 8. MLlib MULTI-VARIATE ANOMALY DETECTION
# -----------------------------
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import FloatType

# Handle any stray nulls
agg_df = agg_df.fillna(0, subset=["avg_dwell", "vessel_count", "weather_score"])

# 1. Assemble features
assembler = VectorAssembler(
    inputCols=["avg_dwell", "vessel_count", "weather_score"],
    outputCol="features_raw"
)
agg_df = assembler.transform(agg_df)

# 2. Scale features (Critical for K-Means)
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(agg_df)
agg_df = scaler_model.transform(agg_df)

# 3. Train K-Means (Learn normal operational states)
kmeans = KMeans(featuresCol="features", k=3, seed=42) 
model = kmeans.fit(agg_df)
agg_df = model.transform(agg_df)

# 4. Calculate Anomaly Score (Distance to Centroid)
centers = model.clusterCenters()
centers_broadcast = spark.sparkContext.broadcast(centers)

@F.udf(returnType=FloatType())
def calculate_distance(features_vec, prediction):
    center = centers_broadcast.value[prediction]
    vec = features_vec.toArray()
    return sum((float(v) - float(c)) ** 2 for v, c in zip(vec, center)) ** 0.5

agg_df = agg_df.withColumn("anomaly_score", calculate_distance(F.col("features"), F.col("prediction")))

# Drop temporary ML columns
agg_df = agg_df.drop("features_raw", "features", "prediction")

print("✅ MLlib Multi-Variate Anomaly Score Computed")

# -----------------------------
# 9. SAVE OUTPUT
# -----------------------------
output_path = "data/processed/fused_features.parquet"

agg_df.write.mode("overwrite").parquet(output_path)

print("🎉 Phase 2 COMPLETE — Features Ready!")

# -----------------------------
# 10. VERIFY
# -----------------------------
spark.read.parquet(output_path).show(5)
