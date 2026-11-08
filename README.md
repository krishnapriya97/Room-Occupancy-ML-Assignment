
# Predicting Room Occupancy Using Environmental Sensor Data

## 1. Project Overview

This project focuses on predicting the number of people present in a room using environmental sensor data.

Machine learning classification algorithms are used to analyze sensor readings such as temperature, light, sound, CO2 and PIR sensor values.

## 2. Dataset Information

- Dataset: Room Occupancy Estimation
- Source: UCI Machine Learning Repository
- Total records: 10,129
- Input features: 16
- Target variable: Room_Occupancy_Count
- Classification type: Multiclass Classification

Dataset link:
https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked dataset information and missing values.
3. Removed Date and Time columns.
4. Separated input features and target variable.
5. Divided the dataset into 80% training and 20% testing data.
6. Applied StandardScaler to the training and testing features.

## 4. Machine Learning Algorithms

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

## 5. Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 6. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9946 | 0.9999 | 0.9947 | 0.9946 | 0.9945 | 0.9834 |
| Decision Tree | 0.9961 | 0.9968 | 0.9961 | 0.9961 | 0.9960 | 0.9880 |
| Random Forest | 0.9970 | 0.9997 | 0.9971 | 0.9970 | 0.9970 | 0.9910 |
| SVM | 0.9970 | 0.9999 | 0.9971 | 0.9970 | 0.9970 | 0.9910 |
| KNN | 0.9951 | 0.9992 | 0.9952 | 0.9951 | 0.9951 | 0.9850 |

## 7. Project Structure

```text
Room_Occupancy_ML_Project/
│
├── ML_Assignment2.ipynb
├── Occupancy_Estimation.csv
├── requirements.txt
├── app.py
├── test_data.csv
├── README.md
│
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── svm.pkl
    ├── knn.pkl
    ├── scaler.pkl
    └── feature_columns.pkl
