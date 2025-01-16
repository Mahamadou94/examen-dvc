import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model():
    X_test = pd.read_csv('/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/X_test_scaled.csv')
    y_test = pd.read_csv('/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/y_test.csv')
    model = joblib.load('/home/ubuntu/Data_scientest/examen-dvc/models/trained_model.pkl')

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': predictions})
    results.to_csv('/home/ubuntu/Data_scientest/examen-dvc/src/data/prediction.csv', index=False)
    
    with open('/home/ubuntu/Data_scientest/examen-dvc/metrics/scores.json', 'w') as f:
        f.write(f'{{"MSE": {mse}, "R2": {r2}}}')

if __name__ == "__main__":
    evaluate_model()
