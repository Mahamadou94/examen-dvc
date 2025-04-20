import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import yaml

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["grid_search"]

def perform_grid_search():
    params = load_params()

    X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv')

    model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train.values.ravel())

    joblib.dump(grid_search.best_params_, 'models/best_params.pkl')

if __name__ == "__main__":
    perform_grid_search()
