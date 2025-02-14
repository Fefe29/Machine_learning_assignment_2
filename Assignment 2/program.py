import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# Load data
def load_data(data_dir):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    return train_df, test_df


# Split dataset
def split_data(train_df):
    X = train_df.iloc[:, :-1]  # Features
    y = train_df.iloc[:, -1]  # Target

    # Split into train (70%), validation (15%), and test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Performance metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} - RSE: {rse:.4f}, R²: {r2:.4f}")

# Train and evaluate linear regression model
def train_evaluate_linear_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)


    calculate_metrics(y_train, y_train_pred, "Train")
    calculate_metrics(y_val, y_val_pred, "Validation")
    calculate_metrics(y_test, y_test_pred, "Test")


# Train and evaluate Ridge regression model
def train_evaluate_ridge_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    param_grid = {'alpha': np.logspace(-3, 3, 10)}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_ridge = grid_search.best_estimator_
    print(f"Best alpha: {grid_search.best_params_['alpha']}")

    # Predictions
    y_train_pred = best_ridge.predict(X_train)
    y_val_pred = best_ridge.predict(X_val)
    y_test_pred = best_ridge.predict(X_test)

    # Performance metrics
    calculate_metrics(y_train, y_train_pred, "Train")
    calculate_metrics(y_val, y_val_pred, "Validation")
    calculate_metrics(y_test, y_test_pred, "Test")


# Train and evaluate Lasso regression model
def train_evaluate_lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    param_grid = {'alpha': np.logspace(-3, 3, 10)}
    lasso = Lasso()
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_lasso = grid_search.best_estimator_
    print(f"Best alpha: {grid_search.best_params_['alpha']}")

    # Predictions
    y_train_pred = best_lasso.predict(X_train)
    y_val_pred = best_lasso.predict(X_val)
    y_test_pred = best_lasso.predict(X_test)

    # Performance metrics
    calculate_metrics(y_train, y_train_pred, "Train")
    calculate_metrics(y_val, y_val_pred, "Validation")
    calculate_metrics(y_test, y_test_pred, "Test")


# Define function for best regression model
def predictCompressiveStrength(Xtest, data_dir):
    train_df, _ = load_data(data_dir)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    best_model = Ridge(alpha=10)  # Example best performing model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(Xtest)
    return y_pred


# Main execution
data_dir = "Data"  # Change this to the correct dataset path
train_df, test_df = load_data(data_dir)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(train_df)
train_evaluate_linear_regression(X_train, X_val, X_test, y_train, y_val, y_test)
train_evaluate_ridge_regression(X_train, X_val, X_test, y_train, y_val, y_test)
train_evaluate_lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test)
