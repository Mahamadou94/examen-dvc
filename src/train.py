import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model():
    X_train = pd.read_csv('/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/y_train.csv')
    best_params = joblib.load('/home/ubuntu/Data_scientest/examen-dvc/models/best_params.pkl')

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train.values.ravel())
    
    joblib.dump(model, '/home/ubuntu/Data_scientest/examen-dvc/models/trained_model.pkl')

if __name__ == "__main__":
    train_model()
