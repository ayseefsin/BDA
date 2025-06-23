from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col

def main():

    spark = SparkSession.builder.appName("KMeansClustering").getOrCreate()
    input_features_path = "gs://bucket-assignment21/Processed-Features.parquet/"
    output_clusters_path = "gs://bucket-assignment21/CustomerSegm.parquet/"
    output_model_path = "gs://bucket-assignment21/ecommerceKMeansModel/"
    data = spark.read.parquet(input_features_path)
    data.cache()
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="cluster", k=3, seed=50)

    model = kmeans.fit(data)

    clustered_data = model.transform(data)

    evaluator = ClusteringEvaluator(featuresCol="scaledFeatures", predictionCol="cluster",
                                    metricName="silhouette", distanceMeasure="squaredEuclidean")

    silhouette = evaluator.evaluate(clustered_data)
    print(f"\nSilhouette Score: {silhouette}")

    centers = model.clusterCenters()
    print("\nScaled Features:")
    for i, center in enumerate(centers):
        print(f"Cluster {i} Center: {center}")

    clustered_data.write.mode("overwrite").parquet(output_clusters_path)
    print(f"\nClustered data saved here: {output_clusters_path}")

    model.write().overwrite().save(output_model_path)
    print(f"K-Means model saved here: {output_model_path}")

    spark.stop()

if __name__ == "__main__":
    main()
