import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_data(file_path):
    """
    Charge les données à partir d'un chemin de fichier, convertit les colonnes de date en datetime,
    extrait les caractéristiques temporelles et supprime la colonne de date originale.
    
    Args:
    file_path (str): Chemin complet du fichier CSV à charger.

    Returns:
    pd.DataFrame: DataFrame avec les colonnes de date traitées.
    """
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])  # Conversion de la colonne 'date' en datetime
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour
    data.drop('date', axis=1, inplace=True)  # Suppression de la colonne 'date'
    return data

def normalize_data(X_train, X_test):
    """
    Normalise les ensembles de données d'entraînement et de test en utilisant MinMaxScaler.

    Args:
    X_train (pd.DataFrame): Données d'entraînement à normaliser.
    X_test (pd.DataFrame): Données de test à normaliser.

    Returns:
    tuple: contient les DataFrames des données d'entraînement et de test normalisées.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    # Chemins des fichiers
    train_path = '/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/X_train.csv'
    test_path = '/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/X_test.csv'

    # Préparation des données
    X_train = prepare_data(train_path)
    X_test = prepare_data(test_path)

    # Normalisation des données
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

    # Sauvegarde des données normalisées
    X_train_scaled.to_csv('/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('/home/ubuntu/Data_scientest/examen-dvc/data/processed_data/X_test_scaled.csv', index=False)

    # Exemple d'affichage des données normalisées
    print(X_train_scaled.head())
    print(X_test_scaled.head())
