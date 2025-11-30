# fraud_detection_project
Healthcare Provider Fraud Detection
Author: Omar Ossama – ID 7001032
Machine Learning – Winter 2025

Project Overview
This project builds a machine learning system to detect fraudulent Medicare healthcare providers. The goal is to analyze multiple healthcare datasets, create provider-level features, handle class imbalance, train models, and evaluate which model is best for identifying high-risk providers.

Dataset
The project uses the Healthcare Provider Fraud Detection Analysis dataset from Kaggle.
Files:

Train_Beneficiarydata.csv

Train_Inpatientdata.csv

Train_Outpatientdata.csv

Train_Labels.csv

Key identifiers:

BeneID for merging beneficiary information

Provider for assigning fraud labels

How to Run the Project

Notebook 1 – Data Preparation
File: 01_data_exploration_and_feature_engineering.ipynb
This notebook:

Loads the four CSV files

Merges inpatient, outpatient, and beneficiary data

Performs basic EDA

Aggregates claim-level data into provider-level features

Saves provider_level_dataset.csv in the data folder

Notebook 2 – Modeling
File: 02_modeling.ipynb
This notebook:

Loads provider_level_dataset.csv

Handles class imbalance using SMOTE and class weights

Trains Logistic Regression, Random Forest, and XGBoost

Runs hyperparameter tuning

Saves final tuned models in the models folder:
logistic_regression_best.joblib
random_forest_best.joblib
xgboost_best.joblib

Notebook 3 – Evaluation
File: 03_evaluation.ipynb
This notebook:

Loads the saved models

Computes evaluation metrics: Precision, Recall, F1, ROC-AUC, PR-AUC

Generates confusion matrices

Plots ROC and Precision-Recall curves

Shows feature importance

Performs error analysis using false positives and false negatives

Project Structure
fraud_detection_project/
data/
models/
notebooks/
01_data_exploration_and_feature_engineering.ipynb
02_modeling.ipynb
03_evaluation.ipynb
reports/
technical_report.pdf
presentation.pptx
README.md

Requirements
Python 3.9+ with the following libraries:
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
joblib

Optional installation command:
pip install -r requirements.txt
