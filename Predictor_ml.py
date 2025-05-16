import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as XGBRegressor
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("pastel")

def load_data(file_path):
    """
    Load the housing dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    print(f"Loading data from {file_path}...")
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(data):
    """
    Perform exploratory data analysis on the housing dataset.
    
    Args:
        data (pandas.DataFrame): The housing dataset
    """
    print("\n===== EXPLORATORY DATA ANALYSIS =====")
    
    # Display basic information about the dataset
    print("\nDataset Information:")
    print(f"Number of records: {data.shape[0]}")
    print(f"Number of features: {data.shape[1]}")
    
    # Display the first few rows of the dataset
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    # Display data types and missing values
    print("\nData types and missing values:")
    missing_data = data.isnull().sum()
    missing_percent = (missing_data / len(data)) * 100
    data_types = data.dtypes
    
    missing_info = pd.DataFrame({
        'Data Type': data_types,
        'Missing Values': missing_data,
        'Missing Percentage': missing_percent.round(2)
    })
    print(missing_info[missing_info['Missing Values'] > 0])
    
    # Display statistical summary of numerical features
    print("\nStatistical summary of numerical features:")
    print(data.describe().T)
    
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot distribution of the target variable (price)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['price'], kde=True)
    plt.title('Distribution of House Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.savefig('plots/price_distribution.png')
    
    # Plot correlation matrix for numerical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 10))
    correlation_matrix = data[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    
    # Plot relationship between key features and price
    key_features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'condition', 'yr_built']
    plt.figure(figsize=(20, 15))
    
    for i, feature in enumerate(key_features):
        plt.subplot(3, 3, i+1)
        sns.scatterplot(x=data[feature], y=data['price'])
        plt.title(f'{feature} vs Price')
        plt.xlabel(feature)
        plt.ylabel('Price')
    
    plt.tight_layout()
    plt.savefig('plots/feature_vs_price.png')
    
    print("\nExploratory data analysis completed. Plots saved in 'plots' directory.")

def preprocess_data(data):
    """
    Preprocess the housing dataset for machine learning.
    
    Args:
        data (pandas.DataFrame): The housing dataset
        
    Returns:
        tuple: X (features), y (target), feature_names (list of feature names)
    """
    print("\n===== DATA PREPROCESSING =====")
    
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Convert date column to datetime and extract useful components
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Create a new feature for house age
    df['house_age'] = df['year'] - df['yr_built']
    
    # Create a feature for renovation status
    df['renovated'] = (df['yr_renovated'] > 0).astype(int)
    
    # Create a feature for total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Create a feature for price per square foot
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    
    # Create a feature for basement presence
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
    
    # Create a feature for lot size category
    df['lot_size_category'] = pd.qcut(df['sqft_lot'], 5, labels=False)
    
    # Extract city from statezip
    df['city'] = df['city'].str.lower()
    
    # Select features for the model
    features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
        'yr_built', 'yr_renovated', 'house_age', 'renovated', 'total_rooms',
        'has_basement', 'lot_size_category', 'month', 'city'
    ]
    
    # Select only the columns we need
    X = df[features].copy()
    y = df['price']
    
    print(f"Selected {len(features)} features for modeling.")
    print(f"Target variable: 'price'")
    
    return X, y, features

def build_model_pipeline(X):
    """
    Build a preprocessing and modeling pipeline.
    
    Args:
        X (pandas.DataFrame): Feature dataframe
        
    Returns:
        sklearn.pipeline.Pipeline: The model pipeline
    """
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Remove 'city' from numerical features if it's there
    if 'city' in numerical_features:
        numerical_features.remove('city')
    
    # Add 'city' to categorical features if it's not there
    if 'city' not in categorical_features and 'city' in X.columns:
        categorical_features.append('city')
    
    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create the full pipeline with the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', None)  # This will be set later
    ])
    
    return model_pipeline

def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple regression models.
    
    Args:
        X (pandas.DataFrame): Feature dataframe
        y (pandas.Series): Target variable
        
    Returns:
        dict: Dictionary of trained models and their performance metrics
    """
    print("\n===== MODEL TRAINING AND EVALUATION =====")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Get the model pipeline
    pipeline = build_model_pipeline(X)
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Set the model in the pipeline
        pipeline.steps[-1] = ('model', model)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Store results
        results[name] = {
            'model': pipeline,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
        
        # Print metrics
        print(f"  Training RMSE: ${train_rmse:.2f}")
        print(f"  Testing RMSE: ${test_rmse:.2f}")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Testing R²: {test_r2:.4f}")
        print(f"  Testing MAE: ${test_mae:.2f}")
    
    # Compare model performance
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Training RMSE': [results[model]['train_rmse'] for model in results],
        'Testing RMSE': [results[model]['test_rmse'] for model in results],
        'Training R²': [results[model]['train_r2'] for model in results],
        'Testing R²': [results[model]['test_r2'] for model in results],
        'Testing MAE': [results[model]['test_mae'] for model in results]
    })
    
    # Sort by testing R²
    model_comparison = model_comparison.sort_values('Testing R²', ascending=False).reset_index(drop=True)
    
    print("\nModel Comparison:")
    print(model_comparison)
    
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Testing R²', data=model_comparison)
    plt.title('Model Comparison - Testing R²')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_comparison_r2.png')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Testing RMSE', data=model_comparison)
    plt.title('Model Comparison - Testing RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_comparison_rmse.png')
    
    # Identify the best model
    best_model_name = model_comparison.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    best_r2 = results[best_model_name]['test_r2']
    best_rmse = results[best_model_name]['test_rmse']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model R²: {best_r2:.4f}")
    print(f"Best Model RMSE: ${best_rmse:.2f}")
    
    return results, best_model

def tune_best_model(X, y, best_model_name, results):
    """
    Tune the hyperparameters of the best model.
    
    Args:
        X (pandas.DataFrame): Feature dataframe
        y (pandas.Series): Target variable
        best_model_name (str): Name of the best model
        results (dict): Dictionary of model results
        
    Returns:
        sklearn.pipeline.Pipeline: The tuned model
    """
    print("\n===== HYPERPARAMETER TUNING =====")
    print(f"Tuning {best_model_name}...")
    
    # Get the best model pipeline
    pipeline = results[best_model_name]['model']
    
    # Define parameter grid based on the best model
    param_grid = {}
    
    if best_model_name == 'Linear Regression':
        # Linear Regression doesn't have hyperparameters to tune
        print("Linear Regression doesn't have hyperparameters to tune.")
        return pipeline
    
    elif best_model_name == 'Ridge Regression':
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    
    elif best_model_name == 'Lasso Regression':
        param_grid = {
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    
    elif best_model_name == 'Random Forest':
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10]
        }
    
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    
    elif best_model_name == 'XGBoost':
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 0.9, 1.0]
        }
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate the tuned model
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Print metrics
    print(f"Tuned model performance:")
    print(f"  Training RMSE: ${train_rmse:.2f}")
    print(f"  Testing RMSE: ${test_rmse:.2f}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Testing MAE: ${test_mae:.2f}")
    
    # Compare with untuned model
    untuned_test_r2 = results[best_model_name]['test_r2']
    untuned_test_rmse = results[best_model_name]['test_rmse']
    
    print(f"\nImprovement after tuning:")
    print(f"  R² improvement: {test_r2 - untuned_test_r2:.4f}")
    print(f"  RMSE improvement: ${untuned_test_rmse - test_rmse:.2f}")
    
    return best_model

def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
    """
    print("\n===== FEATURE IMPORTANCE ANALYSIS =====")
    
    # Get the model from the pipeline
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_step = model.named_steps['model']
    else:
        model_step = model
    
    # Check if the model has feature_importances_ attribute
    if hasattr(model_step, 'feature_importances_'):
        # Get feature importances
        importances = model_step.feature_importances_
        
        # Get feature names after preprocessing
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            if hasattr(preprocessor, 'transformers_'):
                # Get all transformed feature names
                transformed_features = []
                for name, transformer, features in preprocessor.transformers_:
                    if name == 'num':
                        transformed_features.extend(features)
                    elif name == 'cat' and hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                        onehot = transformer.named_steps['onehot']
                        if hasattr(onehot, 'get_feature_names_out'):
                            cat_features = onehot.get_feature_names_out(features)
                            transformed_features.extend(cat_features)
                
                # If the number of transformed features matches the number of importances
                if len(transformed_features) == len(importances):
                    feature_names = transformed_features
        
        # Create a dataframe of feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Print feature importances
        print("Feature Importances:")
        print(feature_importance.head(20))
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        
    elif hasattr(model_step, 'coef_'):
        # For linear models
        coefs = model_step.coef_
        
        # Get feature names after preprocessing
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            if hasattr(preprocessor, 'transformers_'):
                # Get all transformed feature names
                transformed_features = []
                for name, transformer, features in preprocessor.transformers_:
                    if name == 'num':
                        transformed_features.extend(features)
                    elif name == 'cat' and hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                        onehot = transformer.named_steps['onehot']
                        if hasattr(onehot, 'get_feature_names_out'):
                            cat_features = onehot.get_feature_names_out(features)
                            transformed_features.extend(cat_features)
                
                # If the number of transformed features matches the number of coefficients
                if len(transformed_features) == len(coefs):
                    feature_names = transformed_features
        
        # Create a dataframe of coefficients
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(coefs)],
            'Coefficient': coefs
        })
        
        # Sort by absolute coefficient value
        feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)
        
        # Print coefficients
        print("Feature Coefficients:")
        print(feature_importance.head(20))
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(20))
        plt.title('Feature Coefficients')
        plt.tight_layout()
        plt.savefig('plots/feature_coefficients.png')
    
    else:
        print("This model doesn't provide feature importances or coefficients.")

def predict_house_price(model, features):
    """
    Predict the price of a house based on its features.
    
    Args:
        model: Trained model
        features (dict): Dictionary of house features
        
    Returns:
        float: Predicted house price
    """
    # Convert features to a DataFrame
    df = pd.DataFrame([features])
    
    # Make prediction
    predicted_price = model.predict(df)[0]
    
    return predicted_price

def get_user_input():
    """
    Get house features from user input.
    
    Returns:
        dict: Dictionary of house features
    """
    print("\n===== ENTER HOUSE DETAILS =====")
    print("Please enter the following details about the house:")
    
    try:
        # Get numerical inputs
        bedrooms = float(input("Number of bedrooms: "))
        bathrooms = float(input("Number of bathrooms: "))
        sqft_living = float(input("Living space square footage: "))
        sqft_lot = float(input("Lot square footage: "))
        floors = float(input("Number of floors: "))
        waterfront = int(input("Waterfront property (1 for Yes, 0 for No): "))
        view = int(input("View rating (0-4, where 4 is best): "))
        condition = int(input("Condition rating (1-5, where 5 is best): "))
        sqft_above = float(input("Square footage above ground: "))
        sqft_basement = float(input("Square footage of basement: "))
        yr_built = int(input("Year built: "))
        yr_renovated = int(input("Year renovated (0 if never renovated): "))
        
        # Calculate derived features
        current_year = pd.Timestamp.now().year
        house_age = current_year - yr_built
        renovated = 1 if yr_renovated > 0 else 0
        total_rooms = bedrooms + bathrooms
        has_basement = 1 if sqft_basement > 0 else 0
        
        # Get categorical inputs
        city = input("City (e.g., seattle, bellevue): ").lower()
        
        # For lot_size_category, we'll use a simple approach
        # In a real application, this would be derived from the training data distribution
        if sqft_lot < 5000:
            lot_size_category = 0
        elif sqft_lot < 10000:
            lot_size_category = 1
        elif sqft_lot < 20000:
            lot_size_category = 2
        elif sqft_lot < 50000:
            lot_size_category = 3
        else:
            lot_size_category = 4
            
        # Get current month
        month = pd.Timestamp.now().month
        
        # Create and return the features dictionary
        return {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'sqft_above': sqft_above,
            'sqft_basement': sqft_basement,
            'yr_built': yr_built,
            'yr_renovated': yr_renovated,
            'house_age': house_age,
            'renovated': renovated,
            'total_rooms': total_rooms,
            'has_basement': has_basement,
            'lot_size_category': lot_size_category,
            'month': month,
            'city': city
        }
    except ValueError as e:
        print(f"Error: {e}")
        print("Please enter valid numerical values. Using default values instead.")
        
        # Return default values if there's an error
        return {
            'bedrooms': 4,
            'bathrooms': 2.5,
            'sqft_living': 2500,
            'sqft_lot': 8000,
            'floors': 2.0,
            'waterfront': 0,
            'view': 0,
            'condition': 4,
            'sqft_above': 2000,
            'sqft_basement': 500,
            'yr_built': 2000,
            'yr_renovated': 0,
            'house_age': 23,
            'renovated': 0,
            'total_rooms': 6.5,
            'has_basement': 1,
            'lot_size_category': 2,
            'month': pd.Timestamp.now().month,
            'city': 'seattle'
        }

def main():
    """
    Main function to run the house price prediction pipeline.
    """
    print("===== HOUSE PRICE PREDICTION MODEL =====")
    
    # Load the dataset
    file_path = "USA Housing Dataset.csv"
    data = load_data(file_path)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Explore the data
    explore_data(data)
    
    # Preprocess the data
    X, y, feature_names = preprocess_data(data)
    
    # Train and evaluate models
    results, best_model = train_and_evaluate_models(X, y)
    
    # Get the name of the best model
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Testing R²': [results[model]['test_r2'] for model in results]
    })
    best_model_name = model_comparison.sort_values('Testing R²', ascending=False).iloc[0]['Model']
    
    # Tune the best model
    tuned_model = tune_best_model(X, y, best_model_name, results)
    
    # Analyze feature importance
    analyze_feature_importance(tuned_model, feature_names)
    
    print("\n===== HOUSE PRICE PREDICTION =====")
    
    # Get house details from user
    print("Now you can predict the price of a house by entering its details.")
    user_choice = input("Would you like to enter house details? (yes/no): ").lower()
    
    if user_choice == 'yes' or user_choice == 'y':
        # Get house details from user input
        house_features = get_user_input()
        print("\nPredicting house price based on your input...")
    else:
        # Use default example house
        print("\nUsing default example house...")
        house_features = {
            'bedrooms': 4,
            'bathrooms': 2.5,
            'sqft_living': 2500,
            'sqft_lot': 8000,
            'floors': 2.0,
            'waterfront': 0,
            'view': 0,
            'condition': 4,
            'sqft_above': 2000,
            'sqft_basement': 500,
            'yr_built': 2000,
            'yr_renovated': 0,
            'house_age': 23,
            'renovated': 0,
            'total_rooms': 6.5,
            'has_basement': 1,
            'lot_size_category': 2,
            'month': pd.Timestamp.now().month,
            'city': 'seattle'
        }
    
    # Display the house features
    print("\nHouse Features:")
    for feature, value in house_features.items():
        print(f"  {feature}: {value}")
    
    # Predict the house price
    predicted_price = predict_house_price(tuned_model, house_features)
    print(f"\nPredicted house price: ${predicted_price:,.2f}")
    
    print("\nHouse Price Prediction Model completed successfully!")

if __name__ == "__main__":
    main()