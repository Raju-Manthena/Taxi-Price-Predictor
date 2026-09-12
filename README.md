# Taxi Price Predictor

A machine learning regression project that predicts taxi trip fares based on trip distance, duration, passenger count, traffic, weather, and pricing-related features.

## Live Demo

[Try the Taxi Price Predictor](https://taxi-price-predictor-35yjxbhfjho45t372bdf2k.streamlit.app/)

## Overview

This project develops a machine learning model to predict the final price of a taxi trip in USD.

The workflow includes:

- Exploratory data analysis and visualization
- Data cleaning and handling of missing values
- Numerical and categorical feature preprocessing
- Training and comparison of multiple regression models
- Cross-validation for model selection
- Final evaluation on a held-out test set
- Deployment using Streamlit

## Dataset

The project uses the **Taxi Price Prediction** dataset from Kaggle.

The dataset contains trip-related and pricing features such as:

| Feature | Description |
|---|---|
| `Trip_Distance_km` | Distance of the trip in kilometers |
| `Time_of_Day` | Time period when the trip started |
| `Day_of_Week` | Day of the week |
| `Passenger_Count` | Number of passengers |
| `Traffic_Conditions` | Traffic conditions during the trip |
| `Weather` | Weather conditions |
| `Base_Fare` | Fixed starting fare |
| `Per_Km_Rate` | Cost charged per kilometer |
| `Per_Minute_Rate` | Cost charged per minute |
| `Trip_Duration_Minutes` | Trip duration in minutes |
| `Trip_Price` | Final trip fare in USD (target) |

## Machine Learning Workflow

### 1. Data Preparation

- Removed rows with missing target values.
- Split the data into training and test sets using an 80/20 split.
- Numerical features were processed using:
  - Median imputation
  - Standard scaling
- Categorical features were processed using:
  - Most-frequent-value imputation
  - One-hot encoding

### 2. Exploratory Data Analysis

The dataset was explored using:

- Numerical feature distributions
- Boxplots for detecting potential outliers
- Categorical feature distributions
- Trip price distributions across categorical features
- Trip distance vs. trip price visualization
- Correlation analysis

### 3. Models Evaluated

The following regression models were trained and compared:

- SGD Regressor
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Voting Regressor

Three-fold cross-validation was used to compare model performance.

## Final Model Performance

The Random Forest Regressor was selected as the final model based on its cross-validation performance.

| Metric | Score |
|---|---:|
| Cross-Validation RMSE | **10.27** |
| Test RMSE | **11.19** |
| Test MAE | **6.20** |
| Test R² | **0.95** |

The final model achieved a test R² of **0.95**, indicating that it explains approximately 95% of the variance in taxi trip prices.

## Repository Structure

```text
Taxi-Price-Predictor/
│
├── app.py
├── taxi_price_predictor.ipynb
├── taxi_trip_pricing.csv
├── taxi_fare_model.pkl
├── requirements.txt
└── .gitignore
