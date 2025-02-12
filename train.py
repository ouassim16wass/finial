# train.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Charger les données nettoyées
train_df = pd.read_csv("clean_train_reduced.csv")

# Rééchantillonnage avec SMOTE
X = train_df.drop(columns=["SalePrice"])
y = train_df["SalePrice"]

# Normalisation de la variable cible pour SMOTE (découpage en classes)
y_bins = pd.qcut(y, q=3, labels=False)

# Appliquer SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_bins)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialiser les modèles
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Entraînement des modèles après rééchantillonnage
rf_model.fit(X_train_resampled, y_train_resampled)
dt_model.fit(X_train_resampled, y_train_resampled)
ann_model.fit(X_train_resampled, y_train_resampled)

# Sauvegarder les modèles
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(dt_model, 'dt_model.pkl')
joblib.dump(ann_model, 'ann_model.pkl')
