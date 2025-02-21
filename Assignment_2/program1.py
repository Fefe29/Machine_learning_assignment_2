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
    y = train_df.iloc[:, -1]   # Target

    # Split into train (70%), validation (15%), and test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Compute performance metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} - RSE: {rse:.4f}, R²: {r2:.4f}")
    return rse, r2

# QUESTION 1: Train & Evaluate ordinary least squares Regression (Validation Approach)
def train_evaluate_linear_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print("\nValidation Approach Results:")
    rse_train, r2_train = calculate_metrics(y_train, y_train_pred, "Train")
    rse_val, r2_val = calculate_metrics(y_val, y_val_pred, "Validation")
    rse_test, r2_test = calculate_metrics(y_test, y_test_pred, "Test")

    return model, (rse_train, r2_train, rse_val, r2_val, rse_test, r2_test)

# QUESTION 1: Train & Evaluate ordinary least squares Regression (Cross-Validation Approach)
def cross_validate_ols(X, y, k=5):
    model = LinearRegression()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    # Convert MSE to RSE
    rse_scores = np.sqrt(-mse_scores)

    print("\nCross-Validation Approach Results:")
    print(f"Mean RSE: {rse_scores.mean():.4f}, Std RSE: {rse_scores.std():.4f}")
    print(f"Mean R²: {r2_scores.mean():.4f}, Std R²: {r2_scores.std():.4f}")

    return rse_scores.mean(), r2_scores.mean()

# QUESTION 2: Train and Evaluate Ridge Regression with Hyperparameter Tuning
def train_evaluate_ridge_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    param_grid = {'alpha': np.logspace(-5, 5, 10)}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_ridge = grid_search.best_estimator_
    print("train_evaluate_ridge_regression")
    print(f"Best alpha: {grid_search.best_params_['alpha']}")

    # Predictions
    y_train_pred = best_ridge.predict(X_train)
    y_val_pred = best_ridge.predict(X_val)
    y_test_pred = best_ridge.predict(X_test)

    # Performance metrics
    calculate_metrics(y_train, y_train_pred, "Train")
    calculate_metrics(y_val, y_val_pred, "Validation")
    calculate_metrics(y_test, y_test_pred, "Test")

    # Plot performance vs alpha
    plt.figure()
    plt.plot(param_grid['alpha'], grid_search.cv_results_['mean_test_score'])
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('R^2 Score')
    plt.title('Ridge Regression Performance')
    plt.show()

# QUESTION 3: Train and Evaluate Lasso Regression with Hyperparameter Tuning
def train_evaluate_lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    param_grid = {'alpha': np.logspace(-5, 5, 10)}
    lasso = Lasso()
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_lasso = grid_search.best_estimator_
    print("Train_evaluate_lasso_regression")
    print(f"Best alpha: {grid_search.best_params_['alpha']}")

    # Predictions
    y_train_pred = best_lasso.predict(X_train)
    y_val_pred = best_lasso.predict(X_val)
    y_test_pred = best_lasso.predict(X_test)

    # Performance metrics
    calculate_metrics(y_train, y_train_pred, "Train")
    calculate_metrics(y_val, y_val_pred, "Validation")
    calculate_metrics(y_test, y_test_pred, "Test")


    # Plot performance vs alpha
    plt.figure()
    plt.plot(param_grid['alpha'], grid_search.cv_results_['mean_test_score'])
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('R^2 Score')
    plt.title('Lasso Regression Performance')
    plt.show()

# Main execution
data_dir = "Data"  # Update with the correct path
train_df, test_df = load_data(data_dir)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(train_df)

# Question 1: Train OLS using validation approach
ols_model, val_results = train_evaluate_linear_regression(X_train, X_val, X_test, y_train, y_val, y_test)

# Question 1: Train OLS using cross-validation
X_full_train = train_df.iloc[:, :-1]
y_full_train = train_df.iloc[:, -1]
cv_rse, cv_r2 = cross_validate_ols(X_full_train, y_full_train, k=5)

# Compare Validation vs. Cross-Validation Results
print("\n--- Comparison of Validation and Cross-Validation Approaches ---")
print(f"Validation Test RSE: {val_results[4]:.4f}, CV RSE: {cv_rse:.4f}")
print(f"Validation Test R²: {val_results[5]:.4f}, CV R²: {cv_r2:.4f}")
print("----------------------------------------------------")
# Question 2: Run Ridge evaluation
train_evaluate_ridge_regression(X_train, X_val, X_test, y_train, y_val, y_test)
print("----------------------------------------------------")
# Question 3: Run Lasso evaluation
train_evaluate_lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test)
print("----------------------------------------------------")
