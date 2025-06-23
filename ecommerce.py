from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, to_date, hour

def main():
    # This function is responsible for reading the eCommerce dataset, performing data cleaning and transformations, and then writing the processed data back to Google Cloud Storage.
    spark = SparkSession .builder .appName("Ecommerce") .getOrCreate()
    # Initializes or retrieves an existing SparkSession for DataFrame operations.
    input_path = "gs://bucket-assignment21/2019-Nov.csv"
    output_path = "gs://bucket-assignment21/Cleaned-2019-Nov.csv/"
    df = spark.read.option("header", True ).csv(input_path, inferSchema=True)
    # The csv file is read and loaded into a DataFrame (df). The parameter header=True indicates that the first row of the file contains column names, while inferSchema=True enables Spark to automatically detect and design the appropriate data types to each column.

    df = df.withColumn("event_time_ts", to_timestamp(col("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'"))
    df = df.withColumn("event_day_only", to_date(col("event_time_ts")))
    # Extracts only the date (year, month, and day) from the "event_time_ts" column and saves it into a new column named "event_day_only"
    df = df.withColumn("event_hour", hour(col("event_time_ts")))
    # Extracts only the hour information from the "event_time_ts" column and stores it in a new column named "event_hour"
    df = df.withColumn("price", col("price").cast("double"))
    # Converts the data type of the "price" column to "double" (decimal) to enable mathematical operation on its values
    df = df.filter(col("price") >= 0)
    # Filters out and removes rows where the "price" column contains negative values. Negative prices are treated as data errors.
    df = df.dropna(subset=["event_time", "event_type" , "product_id" , "price", "user_id"])
    # Removes rows that contain null values in critical columns: "event_time" , "event_type" , "product_id" , "price", "user_id". These columns are essential for conducting meaningful analysis.
    df = df.fillna({"category_id" : "not available" , "category_code" : "not available" , "brand" : "not available", "user_session" : "not available"})
    # Null values in specific categorical columns are replaced with the string "not available". This helps preserve the dataset structure while clearly indicating missing information.
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
    # Writes the cleaned DataFrame to GCS as a single CSV file by merging partitions (coalesce), overwriting existing files, and including a header row.
    spark.stop()
if __name__ == "__main__": main()
# Calls the main() function when the script is executed directly; allows dual usage as standalone script or importable module.
