import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def perform_grid_search():
    X_train = pd.read_csv('/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/y_train.csv')

    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train.values.ravel())
    
    joblib.dump(grid_search.best_params_, '/home/ubuntu/Data_scientest/examen-dvc/models/best_params.pkl')

if __name__ == "__main__":
    perform_grid_search()
