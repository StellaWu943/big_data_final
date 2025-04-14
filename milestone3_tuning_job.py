
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# Start Spark session
spark = SparkSession.builder.appName("Milestone3_HyperparameterTuning").getOrCreate()

# Load the final processed DataFrame
final_data = spark.read.parquet("final_data.parquet")

# Split into training and testing sets
train_data, test_data = final_data.randomSplit([0.7, 0.3], seed=42)

# Initialize Decision Tree Classifier
dtc = DecisionTreeClassifier(labelCol="LateIndex", featuresCol="features")

# Define evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="LateIndex", predictionCol="prediction", metricName="accuracy"
)

# Define parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(dtc.maxDepth, [3, 5, 10]) \
    .addGrid(dtc.maxBins, [16, 32, 64]) \
    .build()

# TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=dtc,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    trainRatio=0.8,
    parallelism=2
)

# Fit model on full training set
tvs_model = tvs.fit(train_data)

# Record all hyperparameter results
metrics = tvs_model.validationMetrics
param_maps = tvs_model.getEstimatorParamMaps()

print("📊 All Hyperparameter Combinations and Their Accuracy:")

result_lines = []
for i, (params, acc) in enumerate(zip(param_maps, metrics)):
    max_depth = params[dtc.maxDepth]
    max_bins = params[dtc.maxBins]
    line = f"  Combo {i+1}: maxDepth={max_depth}, maxBins={max_bins} --> Accuracy={acc:.4f}"
    print(line)
    result_lines.append(line)

# Get the best model
best_model = tvs_model.bestModel
predictions = best_model.transform(test_data)
accuracy = evaluator.evaluate(predictions)

# Print best parameters and accuracy
print("Best Decision Tree Model Parameters:")
print(f"  - maxDepth: {best_model._java_obj.getMaxDepth()}")
print(f"  - maxBins: {best_model._java_obj.getMaxBins()}")
print(f"Test Set Accuracy on full data: {accuracy:.4f}")

# Save result summary
with open("milestone3_result_summary.txt", "w") as f:
    f.write("All Hyperparameter Combinations and Their Accuracy:\n")
    for line in result_lines:
        f.write(line + "\n")
    f.write("\nBest Decision Tree Model Parameters on FULL DATA:\n")
    f.write(f"  - maxDepth: {best_model._java_obj.getMaxDepth()}\n")
    f.write(f"  - maxBins: {best_model._java_obj.getMaxBins()}\n")
    f.write(f"Test Set Accuracy on full data: {accuracy:.4f}\n")

print("📄 Results saved to 'milestone3_result_summary.txt'")
