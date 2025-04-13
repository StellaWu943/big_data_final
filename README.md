# Enhancing Bus Punctuality: A Machine Learning Approach to Binary Classification of Bus Arrival Times
Group member: Silin Chen, Xinyi Zhang, Stella Wu

## Overview (Silin)
Efficient public transportation is crucial for urban mobility, yet maintaining punctuality remains a significant challenge due to variable urban traffic
conditions. This project focuses on utilizing a detailed dataset from the New York City Metropolitan Transportation Authority (MTA), which includes real-time bus locations, scheduled and actual arrival times, among other features, to develop a binary classification
model. The goal is to accurately classify bus arrivals into two categories: 'On Time' and 'Not On Time'. By applying machine learning techniques to analyze and model patterns based on GPS data and time schedules, the model aims to predict punctuality, thus providing reliable arrival information to passengers and aiding transit authorities in service optimization. This project seeks to demonstrate how machine learning can be effectively used to improve the reliability of public transit systems, enhancing commuter experience and operational efficiency.

## Statement of Problem (Xinyi)


## Data sources (Stella)
The dataset utilized in this project is derived from the New York City Metropolitan Transportation Authority (MTA) bus data stream services. It includes
comprehensive real-time data captured every ten minutes, detailing bus locations, routes, and designated stops across the city. Each entry in the dataset specifies the bus's scheduled and actual arrival times at stops, providing insights into the punctuality of services—whether buses are on time, ahead of schedule, or delayed. This data is sourced from the MTA SIRI (Service Interface for Real Time Information) Real Time data feed and the MTA GTFS
(General Transit Feed Specification) Schedule data, ensuring high reliability and accuracy. The dataset offers a robust framework for analyzing urban transit patterns and optimizing bus scheduling and routing efficiency.
Link to dataset: https://www.kaggle.com/datasets/stoney71/new-york-city-transport-statistics?select=mta_1706.csvjn

## Repository structure (Silin)

## Approach (and any installation instructions) (Xinyi)

## Timeline/Deliverables (Stella)
- Milestone0 ():
- Milestone1 ():
- Milestone2 ():
- Milestone3 ():
- Milestone4 ():

## Resources (Silin)

## How to contribute (if open-source) (Xinyi)

## Sample run/output?

### First model: Decision Tree (Baseline model) (Silin)


#### Feature Engineering (Stella)

#### Hyperparameter tuning (Xinyi)

### Second Model: Random Forest (Xinyi)

### Third Model: Logistic Regression (Silin)

The logistic regression model achieved perfect classification performance, with an accuracy of 100% on both the training and test datasets. This result indicates that the model was able to completely distinguish between the two classes (LateIndex = 0 and 1) without any misclassifications. While this level of accuracy is impressive, it may also suggest potential data leakage or overly simplistic data patterns that the model easily captured. Further evaluation with additional metrics (e.g., ROC-AUC, precision, recall) and validation on unseen data is recommended to confirm the model’s generalizability.

<p align="center" width="100%">
<img src="images\llm_hall.png" alt="" style="width: 60%; min-width: 300px; display: block; margin: auto;">
</p>

### Forth Model: Gradient Bosted Decision Trees (Stella)

### Performance comparison (Stella)



