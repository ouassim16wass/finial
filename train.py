# train.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Charger les données nettoyées
train_df = pd.read_csv("clean_train_reduced.csv")

# Séparation des données
X = train_df.drop(columns=["SalePrice"])
y = train_df["SalePrice"]

# Division des données sans rééchantillonnage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser les modèles
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Entraînement des modèles
rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
ann_model.fit(X_train, y_train)

# Sauvegarder les modèles
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(dt_model, 'dt_model.pkl')
joblib.dump(ann_model, 'ann_model.pkl')
