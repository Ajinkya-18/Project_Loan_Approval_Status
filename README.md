# Loan Approval Prediction

This project is designed to predict the likelihood of loan approval using a dataset from the Kaggle Playground Series (Season 4, Episode 10). The notebook walks through data preprocessing, exploratory data analysis, and modeling steps using Python libraries such as pandas, seaborn, and scikit-learn.

## 📁 Dataset

**Kaggle Dataset Link**: [Playground Series - Season 4, Episode 10] (https://www.kaggle.com/competitions/playground-series-s4e10/data)

The dataset includes various features such as:
- Personal information (income, age, employment)
- Credit-related details (loan amount, credit score, past defaults)
- Loan-specific data (purpose, grade, interest rate)

## 🚀 Project Structure

1. **Data Loading and Inspection**
2. **Handling Missing Values**
3. **Feature Engineering**
4. **Encoding Categorical Variables**
5. **Model Training**
6. **Evaluation Metrics**

## 📦 Libraries Used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## 📊 Feature Encoding Strategy
- One-Hot Encoding
- Ordinal Encoding
- Label Encoding

## ML Models used:
- RandomForestClassifier (RFC)
- HistGradientBoostingClassifier (HGBC)
- KNeighborsClassifier (KNC)
- Logistic Regression
- Linear Support Vector Classifier
  
KNC, HGBC, and RFC performed well with over 90 % accuracies with HGBC performing the best with accuracy of over 93%.

The dataset also contains a test.csv file which can be used to generate predictions on new data (deployment simulation).


#### Author: 
Ajinkya Tamhankar
