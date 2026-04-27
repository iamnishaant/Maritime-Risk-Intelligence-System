from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, current_timestamp, lit, date_trunc
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# -----------------------------
# 1. Start Spark Session
# -----------------------------
spark = SparkSession.builder \
    .appName("AIS_Phase0_1") \
    .getOrCreate()

# -----------------------------
# 2. Define Schema (CRITICAL)
# -----------------------------
ais_schema = StructType([
    StructField("mmsi", StringType(), True),
    StructField("base_date_time", StringType(), True),
    StructField("lat", DoubleType(), True),
    StructField("lon", DoubleType(), True),
    StructField("sog", DoubleType(), True)
])

# -----------------------------
# 3. Load Data (Distributed)
# -----------------------------
df = spark.read \
    .option("header", True) \
    .schema(ais_schema) \
    .csv("../data/raw/*.csv")

print("Raw Data Loaded")

# -----------------------------
# 4. Rename Columns
# -----------------------------
df = df.withColumnRenamed("mmsi", "vessel_id") \
       .withColumnRenamed("base_date_time", "timestamp") \
       .withColumnRenamed("lat", "latitude") \
       .withColumnRenamed("lon", "longitude") \
       .withColumnRenamed("sog", "speed")

# -----------------------------
# 5. Convert Types + Validate
# -----------------------------
df = df.withColumn("timestamp", to_timestamp(col("timestamp")))

df = df.dropna(subset=[
    "vessel_id", "timestamp", "latitude", "longitude", "speed"
])

print("Validation Done")

# -----------------------------
# 6. Add Metadata (BIG DATA)
# -----------------------------
df = df.withColumn("source", lit("AIS")) \
       .withColumn("ingestion_time", current_timestamp())

# -----------------------------
# 7. Time Standardization
# -----------------------------
df = df.withColumn("time", date_trunc("hour", col("timestamp")))

# -----------------------------
# 8. Basic Filtering (optional)
# -----------------------------
df = df.filter(col("speed") >= 0)

# -----------------------------
# 9. Save as Parquet
# -----------------------------
output_path = "../data/processed/clean_ais.parquet"

df.write.mode("overwrite").parquet(output_path)

print("Saved clean data to Parquet")

# -----------------------------
# 10. Verify
# -----------------------------
df_check = spark.read.parquet(output_path)
df_check.show(5)
