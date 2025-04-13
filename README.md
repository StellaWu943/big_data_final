# Enhancing Bus Punctuality: A Machine Learning Approach to Binary Classification of Bus Arrival Times
Group member: Silin Chen, Xinyi Zhang, Stella Wu

## Overview (Silin)
Efficient public transportation is crucial for urban mobility, yet maintaining punctuality remains a significant challenge due to variable urban traffic conditions. This project focuses on utilizing a detailed dataset from the New York City Metropolitan Transportation Authority (MTA), which includes real-time bus locations, scheduled and actual arrival times, among other features, to develop a binary classification model. The goal is to accurately classify bus arrivals into two categories: 'On Time' and 'Not On Time'. By applying machine learning techniques to analyze and model patterns based on GPS data and time schedules, the model aims to predict punctuality, thus providing reliable arrival information to passengers and aiding transit authorities in service optimization. This project seeks to demonstrate how machine learning can be effectively used to improve the reliability of public transit systems, enhancing commuter experience and operational efficiency.

## Statement of Problem (Xinyi)
Urban transit systems are complex, and maintaining consistent bus arrival times in a city like New York presents daily operational challenges. Factors such as unpredictable traffic conditions, passenger volume fluctuations, and weather disruptions contribute to late arrivals. For transit authorities and passengers alike, bus delays are a major inconvenience, undermining public trust in the reliability of transportation services.

Our project aims to address this problem through a machine learning-based solution: building a predictive classification model to determine whether a bus will arrive on time. By leveraging large-scale, real-time bus data from the MTA, we attempt to capture underlying spatial-temporal patterns in vehicle movement and scheduled adherence. This formulation as a binary classification task enables us to support transit agencies in identifying high-risk delays early and to eventually inform tools like live bus tracking and performance dashboards.

## Data sources (Stella)
The dataset utilized in this project is derived from the New York City Metropolitan Transportation Authority (MTA) bus data stream services. It includes
comprehensive real-time data captured every ten minutes, detailing bus locations, routes, and designated stops across the city. Each entry in the dataset specifies the bus's scheduled and actual arrival times at stops, providing insights into the punctuality of services—whether buses are on time, ahead of schedule, or delayed. This data is sourced from the MTA SIRI (Service Interface for Real Time Information) Real Time data feed and the MTA GTFS
(General Transit Feed Specification) Schedule data, ensuring high reliability and accuracy. The dataset offers a robust framework for analyzing urban transit patterns and optimizing bus scheduling and routing efficiency.
Link to dataset: https://www.kaggle.com/datasets/stoney71/new-york-city-transport-statistics?select=mta_1706.csvjn

## Repository structure (Silin)

## Approach (and any installation instructions) (Xinyi)
This project was developed entirely on Google Cloud Platform (GCP) using Dataproc clusters and JupyterLab for scalable, distributed data processing and machine learning. Given the volume and velocity of real-time bus data (over 5GB), GCP provided the computational resources and flexibility needed to efficiently run PySpark workflows and tune models at scale.

Our approach followed a structured pipeline:
1. Data preprocessing: Raw data was cleaned, filtered, and transformed using PySpark DataFrame operations.
2. Feature engineering: We used VectorAssembler to combine numerical features such as latitude/longitude, distance to stop, and scheduled timestamps. Categorical variables like direction were encoded using StringIndexer and OneHotEncoder.
3. Model training and tuning: We implemented and evaluated four models using PySpark MLlib: Decision Tree, Logistic Regression, Random Forest, and Gradient Boosted Trees. Hyperparameters were tuned using TrainValidationSplit.
4. Evaluation and visualization: Model performance was evaluated on training and testing splits using accuracy, and key results were saved or visualized directly in GCP JupyterLab.

To reproduce the pipeline:
Provision a Dataproc Cluster with Spark 3.5.0 and Python 3.7+.
Launch a JupyterLab notebook server through GCP.
Upload the milestone notebooks and data files to the Dataproc environment.
Run the notebooks sequentially from milestone 2 to milestone 4, using PySpark to process and model the data.

All necessary dependencies (e.g., pyspark, numpy, matplotlib) come pre-installed with Dataproc. No additional packages were required outside of the standard Spark MLlib ecosystem. Our GCP-based workflow ensured scalability, reproducibility, and seamless resource management across all phases of the project.

## Timeline/Deliverables (Stella)
- Milestone0 ():
- Milestone1 ():
- Milestone2 ():
- Milestone3 ():
- Milestone4 ():

## Resources (Silin)

- [Project Repository](https://github.com/StellaWu943/big_data_final): Access the milestones jupyter notebook.

This project was developed using PySpark and related tools from the Apache Spark ecosystem. The following resources were instrumental in building and evaluating the machine learning models:

- [PySpark MLlib Documentation](https://arxiv.org/abs/2305.14292](https://spark.apache.org/docs/latest/ml-guide.html)): Official guide for Spark’s machine learning API.
- [PySpark API Reference](https://arxiv.org/abs/2305.14292](https://spark.apache.org/docs/latest/ml-guide.html)): Include syntax and usage of ML transformers, estimators, and evaluators.
- [Apache Spark 3.5.0 Release Notes](https://arxiv.org/abs/2305.14292](https://spark.apache.org/releases/spark-release-3-5-0.html)): For understanding recent changes, including deprecations (e.g., Arrow config changes).

## How to contribute (if open-source) (Xinyi)
While this project was designed as a course deliverable, its pipeline structure and focus on transportation performance modeling make it suitable for further development as an open-source research tool.

Future contributions could include:
Incorporating real-time weather and traffic feeds as additional features;
Building a web-based interface for real-time arrival predictions;
Extending support for multi-class classification (e.g., early, on time, delayed);
Applying model interpretability tools (e.g., SHAP, feature importance analysis) to understand delay drivers.

If open-sourced, contributors are encouraged to fork the repository, follow the existing structure, and submit pull requests with documentation.

## Sample run/output?

### First model: Decision Tree (Baseline model) (Silin)

The first model implemented in this milestone is a Decision Tree Classifier, chosen as the baseline for evaluating classification performance. It was trained on engineered features derived from NYC MTA bus GPS data, such as route direction, geographic coordinates, vehicle distance to the stop, and lateness indicators.

**Model Details:**
Algorithm: DecisionTreeClassifier from pyspark.ml.classification

Features: DirectionRef, origin/destination latitude & longitude, distance from stop, vehicle location

Label: LateIndex (binary classification: 1 = late, 0 = on-time)

Split: 80% training / 20% testing

**Model Performance:**

Training Accuracy: 100.00%

Test Accuracy: 100.00%

<p align="center" width="100%">
<img src="images\dtc_acc.png" alt="" style="width: 60%; min-width: 300px; display: block; margin: auto;">
</p>

These results indicate that the model perfectly classified the outcome variable for both seen and unseen data. While this level of performance appears ideal in theory, achieving 100% accuracy on both the training and test sets is highly unusual in real-world applications. Such outcomes may point to the presence of highly separable patterns in the dataset, which can make classification trivially easy. However, it can also raise concerns about potential overfitting, where the model memorizes training data instead of learning generalizable patterns. Another possible explanation is data leakage or the inclusion of redundant features that directly or indirectly reveal the label.

To ensure that the model is truly robust and generalizable, additional evaluation is recommended. This includes using cross-validation to assess performance consistency across different data splits and inspecting feature importance to identify and remove any redundant or overly influential features. It is also advisable to compare the Decision Tree model against more regularized or ensemble-based alternatives, such as Random Forests or Gradient-Boosted Trees, which are generally more resistant to overfitting and better suited for capturing complex patterns in the data.


#### Feature Engineering (Stella)

#### Hyperparameter tuning (Xinyi)
Hyperparameter tuning was conducted on a Decision Tree Classifier to assess the impact of model configuration on predictive performance. The goal was to identify the most effective combination of `maxDepth` and `maxBins`—two of the most influential parameters for tree-based models.

We adopted a two-step tuning strategy. First, to ensure our tuning logic was correctly implemented, we conducted an initial search on a 10% subset of the training data. A grid of 9 parameter combinations was tested using PySpark’s `TrainValidationSplit`, with `maxDepth` values set to [3, 5, 10] and `maxBins` values to [16, 32, 64]. The evaluation metric was classification accuracy.

Once the pipeline was verified, we scaled the tuning process to the full training dataset. All combinations were retrained, and accuracy scores were recorded for each. The best performing model had:
- **maxDepth = 5**
- **maxBins = 16**
- **Test Accuracy = 62.35%**

A full list of the tested hyperparameter combinations and their corresponding accuracy scores was saved to `milestone3_result_summary.txt`. These experiments demonstrated that moderately complex trees generalized better than very deep or shallow ones. It also emphasized the importance of using scalable tuning methods like `TrainValidationSplit` in big data environments such as Google Cloud Platform, where distributed computation significantly reduced model selection time.

### Second Model: Random Forest (Xinyi)
As our second model, we implemented a Random Forest Classifier using PySpark’s RandomForestClassifier API. This ensemble method was chosen to address potential overfitting observed in the Decision Tree model and to better capture nonlinear patterns in the data.

We retained the same input feature set as the baseline model for comparability, which includes spatial coordinates, vehicle proximity to stop, and route direction information. A StringIndexer was used to convert the binary label Late into a numerical class LateIndex.

Training Configuration:
	•	Features: DirectionRef, OriginLat, OriginLong, DestinationLat, DestinationLong, VehicleLocation_Latitude, VehicleLocation_Longitude, DistanceFromStop
	•	Label: LateIndex
	•	Data Split: 70% training / 30% testing
	•	Hyperparameters: numTrees = 20, maxDepth = 5

Model Performance:
	•	Training Accuracy: 100.00%
	•	Test Accuracy: 100.00%

These results matched those of the Decision Tree and Logistic Regression models. While this perfect accuracy may suggest exceptionally clean or highly predictive features, it also raises questions about potential label leakage or data redundancy. In future iterations, we recommend holding out a subset of the data or integrating temporal segmentation (e.g., rush hours vs. non-peak) to evaluate the generalizability of the model.

A screenshot of the Random Forest results is included in the appendix (see rf_acc.png).
<img width="672" alt="rf_acc" src="https://github.com/user-attachments/assets/5ccb0a90-7b60-4cdf-b104-b278cc2f7b64" />


### Third Model: Logistic Regression (Silin)

The logistic regression model achieved perfect classification performance, with an accuracy of 100% on both the training and test datasets. This result indicates that the model was able to completely distinguish between the two classes (LateIndex = 0 and 1) without any misclassifications. While this level of accuracy is impressive, it may also suggest potential data leakage or overly simplistic data patterns that the model easily captured. Further evaluation with additional metrics (e.g., ROC-AUC, precision, recall) and validation on unseen data is recommended to confirm the model’s generalizability.

<p align="center" width="100%">
<img src="images\lr_acc.png" alt="" style="width: 60%; min-width: 300px; display: block; margin: auto;">
</p>

### Forth Model: Gradient Bosted Decision Trees (Stella)

### Performance comparison (Stella)



