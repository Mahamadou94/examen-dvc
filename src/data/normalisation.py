import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["normalisation"]

def prepare_data(data):
    data['date'] = pd.to_datetime(data['date'])  # Conversion en datetime
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour
    data.drop('date', axis=1, inplace=True)  # Suppression de la colonne originale
    return data

def normalize_data(X_train, X_test, method="minmax"):
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)

if __name__ == "__main__":
    params = load_params()
    method = params["scaler"]

    # Chargement initial
    X_train = pd.read_csv('data/processed_data/X_train.csv')
    X_test = pd.read_csv('data/processed_data/X_test.csv')

    # Préparation des données avec la gestion des dates
    X_train = prepare_data(X_train)
    X_test = prepare_data(X_test)

    # Normalisation des données préparées
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, method)

    # Sauvegarde des données normalisées
    X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv', index=False)

    print("✅ Normalisation réussie.")
