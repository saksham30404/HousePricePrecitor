# House Price Predictor

## Overview
This project builds a machine learning model to predict house prices based on the USA Housing Dataset. It implements various regression algorithms to find the most accurate prediction model and includes comprehensive data analysis, preprocessing, and model evaluation.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Project Structure
- `USA Housing Dataset.csv` - The dataset containing housing information
- `Predictor_ml.py` - The main Python script that builds and evaluates the prediction models
- `plots/` - Directory containing visualizations generated during analysis (created when running the script)

## Dataset
The USA Housing Dataset contains information about houses including:
- Price (target variable)
- Number of bedrooms and bathrooms
- Square footage (living space, lot, above ground, basement)
- Number of floors
- Waterfront status
- View rating
- Condition rating
- Year built and renovated
- Location information (street, city, state)

## Features
The model uses both original features from the dataset and engineered features:

### Original Features
- Bedrooms
- Bathrooms
- Square footage (living, lot, above ground, basement)
- Floors
- Waterfront
- View
- Condition
- Year built
- Year renovated
- City

### Engineered Features
- House age (current year - year built)
- Renovation status (binary)
- Total rooms (bedrooms + bathrooms)
- Basement presence (binary)
- Lot size category (binned)
- Month of listing

## Models
The project evaluates several regression models:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regression
5. Gradient Boosting Regression
6. XGBoost Regression

The best performing model is selected based on R² score and RMSE (Root Mean Square Error), then further optimized through hyperparameter tuning.

## Installation
To run this project, you need Python 3.6+ and the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage
1. Clone this repository
2. Ensure the dataset file `USA Housing Dataset.csv` is in the project root directory
3. Run the main script:

```bash
python Predictor_ml.py
```

The script will:
1. Load and explore the dataset
2. Preprocess the data and engineer features
3. Train multiple regression models
4. Evaluate and compare model performance
5. Tune the best performing model
6. Analyze feature importance
7. Provide an example prediction

## Results
The script generates several visualizations in the `plots/` directory:
- Distribution of house prices
- Correlation matrix of numerical features
- Feature vs. price relationships
- Model comparison (R² and RMSE)
- Feature importance/coefficients

Performance metrics for each model include:
- Training and testing RMSE (Root Mean Square Error)
- Training and testing R² (coefficient of determination)
- Testing MAE (Mean Absolute Error)

## How It Works

### Data Preprocessing
The script performs several preprocessing steps:
1. Handles missing values
2. Converts date information to useful features
3. Creates engineered features
4. Normalizes numerical features
5. Encodes categorical features

### Model Pipeline
The model uses a scikit-learn pipeline that includes:
1. Preprocessing (imputation, scaling, encoding)
2. Model training
3. Prediction

### Hyperparameter Tuning
The best model undergoes hyperparameter tuning using GridSearchCV to find the optimal configuration.

### Feature Importance
The script analyzes which features have the most impact on price predictions, helping to understand the key factors in house pricing.

## Interactive Prediction
The script allows you to interactively predict house prices by:
1. Entering your own house details when prompted
2. Using default values if you prefer not to enter details

You'll be asked to provide information such as:
- Number of bedrooms and bathrooms
- Square footage (living space, lot, above ground, basement)
- Number of floors
- Waterfront status and view rating
- Condition rating
- Year built and renovated
- City location

The model will then calculate derived features (like house age and renovation status) and predict the house price based on your input.

## Contributing
Contributions to improve the model or extend its functionality are welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Created by Saksham Arora