import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import yaml
import json

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["evaluate"]["metrics"]

def evaluate_model():
    X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed_data/y_test.csv')
    model = joblib.load('models/trained_model.pkl')

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics_to_save = load_params()
    scores = {}
    if "mse" in metrics_to_save:
        scores["MSE"] = mse
    if "r2" in metrics_to_save:
        scores["R2"] = r2

    # sauvegarder prediction
    results = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': predictions})
    results.to_csv('src/data/prediction.csv', index=False)

    # sauvegarder scores.json
    with open('metrics/scores.json', 'w') as f:
        json.dump(scores, f)

    print("✅ Évaluation terminée avec succès.")

if __name__ == "__main__":
    evaluate_model()
