# evaluate.py
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Charger les données nettoyées
test_df = pd.read_csv("clean_train_reduced.csv")

# Séparation des données
X = test_df.drop(columns=["SalePrice"])
y = test_df["SalePrice"]

# Charger les modèles sauvegardés
rf_model = joblib.load('rf_model.pkl')


# Prédictions avec chaque modèle
rf_preds = rf_model.predict(X)


# Évaluation des modèles
rf_mae = mean_absolute_error(y, rf_preds)


rf_mse = mean_squared_error(y, rf_preds)


# Affichage des résultats
print(f"Random Forest MAE: {rf_mae}, MSE: {rf_mse}")
