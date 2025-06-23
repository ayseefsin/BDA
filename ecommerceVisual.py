import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, when, round, avg, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import VectorUDT
import pandas as pd
import io
from google.cloud import storage

def main():
    spark = SparkSession.builder.appName("KMeansVisualization").getOrCreate()
    input_segments_path = "gs://bucket-assignment21/CustomerSegm.parquet/"
    output_gcs_bucket = "bucket-assignment21"
    feature_names = [
        "total_purchases",
        "total_views",
        "total_carts",
        "distinct_products_interacted" ,
        "total_sessions",
        "avg_purchase_price",
        "conversion_rate"
    ]


    try:
        clustered_df = spark.read.parquet(input_segments_path)
        clustered_df.cache()
        print("Clustered Data read successfully.")
        clustered_df.printSchema()
    except Exception as e:
        print(e)
        spark.stop()
        return
    print("\nChart 1: Number of Users in Each Cluster is generated...")
    cluster_counts_pd = clustered_df.groupBy("cluster").agg(count("user_id").alias("user_count")).toPandas()
    plt.figure(figsize = (9,9))
    plt.pie(cluster_counts_pd['user_count'], labels = cluster_counts_pd['cluster'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    plt.title("Distribution of Number of Users in Each Cluster")
    plt.axis('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    client =storage.Client()
    bucket = client.get_bucket(output_gcs_bucket)
    blob = bucket.blob(os.path.join(output_gcs_folder, 'users_per_cluster_pie_chart.png'))
    blob.upload_from_file(buf, content_type='image/png')
    print(f"Chart 'users_per_cluster_pie_chart.png' saved: gs://{output_gcs_bucket}/{output_gcs_folder}")
    plt.close()
    print("\nChart 2: Generating Cluster Centers comparison...")
    vector_size = len(feature_names)
    @udf(ArrayType(DoubleType()))
    def vector_to_array(v: VectorUDT) -> list:
        if v is None:
            return [0.0] * vector_size
        return v.tolist().tolist()
    clustered_df = clustered_df.withColumn("denseScaledFeatures", vector_to_array(col("scaledFeatures")))

    avg_scaled_features = clustered_df.groupby("cluster").agg(*[avg(col("denseScaledFeatures")[i]).alias(f"{feature_names[i]}_avg_scaled") for i in range(vector_size)]).orderBy("cluster").toPandas()
    avg_scaled_features_melted = avg_scaled_features.melt(id_vars=['cluster'], var_name='feature', value_name="avg_scaled_value")
    plt.figure(figsize=(14, 7))

    sns.barplot(x='feature', y='avg_scaled_value', hue='cluster', data=avg_scaled_features_melted, palette='viridis')
    plt.title('Average Scaled Feature Values by Cluster')
    plt.xlabel('Feature')
    plt.ylabel('Average Scaled Value')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Cluster')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    blob = bucket.blob(os.path.join(output_gcs_folder, 'cluster_centers_bar_chart.png'))
    blob.upload_from_file(buf, content_type='image/png')
    print(f"Grafik 'cluster_centers_bar_chart.png' saved: gs://{output_gcs_bucket}/{output_gcs_folder}")
    plt.close()
    spark.stop()
    print("\nvisualisation done. charts are saved.")
if __name__ == "__main__":
    main()
