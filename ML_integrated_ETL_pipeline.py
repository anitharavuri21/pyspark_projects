from pyspark.sql import SparkSession
from pyspark.sql.functions import count,col,when,mean,unix_timestamp,round
import matplotlib.pyplot as plt
import pandas
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import seaborn as sns

spark=SparkSession.builder.appName('ETL pipeline').getOrCreate()

# extracting dataset
df=spark.read.options(header='True',inferSchema='True').parquet("C:/Users/anithalavanya/Downloads/yellow_tripdata_2024-01.parquet")

# finding null values in the dataset
null_cols=df.select([count(when(col(c).isNull(),c)).alias(c) for c in df.columns])
null_cols=null_cols.select([col(c) for c in null_cols.columns if null_cols.first()[c]>0])
null_cols.show()

# filling the missing data
pc_mean=df.select(mean(col('passenger_count'))).first()[0]
af_mean=df.select(mean(col('Airport_fee'))).first()[0]
cs_mean=df.select(mean(col('congestion_surcharge'))).first()[0]

df_filled=df.na.fill({
    'passenger_count':pc_mean,
    'RatecodeID':0,
    'store_and_fwd_flag':'N',
    'congestion_surcharge':cs_mean,
    'Airport_fee':af_mean
})

# checking for null values again after handling the missing data
df_filled.select([count(when(col(c).isNull(),c)).alias(c) for c in df.columns]).show()

# visualizing the dataset for checking anomalies
pandas_df=df_filled.limit(50000).toPandas()
plt.scatter(pandas_df['passenger_count'],pandas_df['total_amount'],alpha=0.5,color='blue')
plt.xlabel('passenger count')
plt.ylabel('total fare')
plt.title('scatter plot: passengers vs fare')
plt.axhline(0, color='red', linestyle='dashed')
plt.grid('True')
plt.show()

plt.scatter(pandas_df['trip_distance'],pandas_df['total_amount'],alpha=0.5,color='blue')
plt.title('Scatter plot: distance vs amount')
plt.xlabel('trip distance')
plt.ylabel('amount')
plt.axhline(0,color='red',linestyle='dashed')
plt.show()

# handling the rows which have the negative values for total_amount, Passenger_count and
# trip_distance as they are not relevant
df_cleaned=df_filled.filter((col('total_amount')>0) & (col('passenger_count')>0))
print(df_filled.count())
print(df_cleaned.count())
df_cleaned=df_cleaned.filter(col('trip_distance')>0)

# transforming the data by adding some meaningful columns
df_transformed=df_cleaned.withColumn('pickup_date',col('tpep_pickup_datetime').cast('date')).withColumn('dropoff_date',col('tpep_dropoff_datetime').cast('date'))
df_transformed = df_transformed.withColumn("fare_per_passenger",col("total_amount") / col("passenger_count"))
df_transformed = df_transformed.withColumn('price_per_mile',round(col('total_amount')/col('trip_distance'),2))
df_transformed = df_transformed.withColumn("trip_duration_min", round((unix_timestamp("tpep_dropoff_datetime")-unix_timestamp("tpep_pickup_datetime"))/60,2))
df_transformed = df_transformed.withColumn("trip_speed_mph", round(col("trip_distance") / (col("trip_duration_min") / 60),2))

# assembling the input features on which the output label is depending for training the model to predict the total_amount
feature_columns=["trip_distance", "fare_amount","extra","mta_tax","tip_amount", "tolls_amount", "improvement_surcharge","congestion_surcharge","Airport_fee" , "price_per_mile","PULocationID","DOLocationID"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
features_vector_df = assembler.transform(df_transformed)

# splitting the dataset into training data and test data
train_df, test_df = features_vector_df.randomSplit([0.8, 0.2], seed=42)

# creating the linear regression model
lr = LinearRegression(featuresCol='features', labelCol='total_amount',regParam=0.1)

# training the model on the training dataset
lr_model = lr.fit(train_df)

# applying the trained model to test data to generate predictions
predictions = lr_model.transform(test_df)
print(predictions.select("total_amount","prediction").show(50))

# Evaluating the results
evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")

evaluator_mae = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)
print(f"Mean Absolute Error (MAE) on test data: {mae}")

evaluator_r2 = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print(f"R-squared (R²) on test data: {r2}")

# visualising the results
predictions_pandas = predictions.select("total_amount", "prediction").toPandas()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=predictions_pandas['total_amount'], y=predictions_pandas['prediction'], alpha=0.3)
plt.xlabel('Actual Total Amount')
plt.ylabel('Predicted Total Amount')
plt.title('Actual vs Predicted Total Amount')
plt.plot(predictions_pandas['total_amount'], predictions_pandas['total_amount'],color='red', linestyle='--', linewidth=2)
plt.show()

# saving the model for later use to avoid training the model again
lr_model.write().overwrite().save('trained_model')

# loading the transformed data to current directory
predictions.write.parquet("predictions.parquet")
