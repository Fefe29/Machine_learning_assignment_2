import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

def predictCompressiveStrength(Xtest, data_dir):
    # Load datasets
    train_data = pd.read_csv(f"{data_dir}/train.csv")
    test_data = pd.read_csv(f"{data_dir}/test.csv")
    
    # Assuming the last column is the target variable (compressive strength)
    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    


    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Xtest_scaled = scaler.transform(Xtest)

    # Split into train (70%), validation (15%), and test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    

    # Model selection using GridSearchCV
    models = {
        "RandomForest": RandomForestRegressor(),
        "SVR": SVR()
    }
    
    params = {
        "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "SVR": {"kernel": ["linear", "rbf"], "C": [1, 10, 100]}
    }
    
    best_model = None
    best_score = -np.inf
    
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, params[model_name], cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        
        score = r2_score(y_val, grid_search.best_estimator_.predict(X_val))
        
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
    
    # Train best model on full dataset
    best_model.fit(X_scaled, y)
    
    # Make predictions on Xtest
    y_pred = best_model.predict(Xtest_scaled)
    
    return y_pred

Xtest=pd.read_csv("Data/test.csv") # a remplacer par le chemin du fichier de test
print(predictCompressiveStrength(Xtest, data_dir="Data"))