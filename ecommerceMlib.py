from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, sum, avg, round, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.window import Window

def main():

    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

    input_path = "gs://bucket-assignment21/Cleaned-2019-Nov.csv/"

    output_path = "gs://bucket-assignment21/Processed-Features.parquet/"

    df = spark.read.option("header", True).csv(input_path, inferSchema=True)

    df.cache()

    user_event_counts = df.groupBy("user_id").agg(
        sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),
        sum(when(col("event_type") == "view", 1).otherwise(0)).alias("total_views"),
        sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("total_carts"),
        countDistinct(col("product_id")).alias("distinct_products_interacted"),
        countDistinct(col("user_session")).alias("total_sessions")
    )

    user_avg_price = df.filter(col("event_type") == "purchase").groupBy("user_id").agg(
        round(avg("price"), 2).alias("avg_purchase_price")
    )

    user_conversion_rate = user_event_counts.withColumn(
        "conversion_rate",
        when(col("total_views") > 0, col("total_purchases").cast("double") / col("total_views")).otherwise(0.0)
    ).select("user_id", "conversion_rate")

    features_df = user_event_counts
    features_df = features_df.join(user_avg_price, on="user_id", how="left")
    features_df = features_df.fillna({"avg_purchase_price": 0.0})
    features_df = features_df.join(user_conversion_rate, on="user_id", how="left")
    features_df = features_df.fillna({"conversion_rate": 0.0})

    feature_columns = [
        "total_purchases",
        "total_views",
        "total_carts",
        "distinct_products_interacted",
        "total_sessions",
        "avg_purchase_price",
        "conversion_rate"
    ]

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
    vectorized_df = assembler.transform(features_df)
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=False)
    scaler_model = scaler.fit(vectorized_df)
    scaled_df = scaler_model.transform(vectorized_df)

    print("After Feature Engineering DataFrame schema:")
    scaled_df.printSchema()
    print("After Feature Engineering DataFrame Sample Data:")
    scaled_df.select("user_id", "features", "scaledFeatures").show(5, truncate=False)
    scaled_df.write.mode("overwrite").parquet(output_path)

    spark.stop()

if __name__ == "__main__":
    main()
