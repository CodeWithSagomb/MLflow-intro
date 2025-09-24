

import mlflow
# 
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

mlflow.set_experiment("Iris Model Experiment")



# autologging_pipeline.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Activate autologging
mlflow.sklearn.autolog(log_model_signatures=False)

def train_and_log_model():
 
    
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    
    n_estimators = 180 
    max_depth = 9

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Précision : {accuracy:.2f}")

if __name__ == "__main__":
    print("Hello from mlflow-intro!")
    print("--- Getting start with autologging ---")
    train_and_log_model()