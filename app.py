from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import os

app = Flask(__name__)  # Correction ici

# 🔹 Configuration de la base PostgreSQL
DB_USER = "myuser"  # Remplace par ton utilisateur PostgreSQL
DB_PASSWORD = "mypassword"  # Remplace par ton mot de passe PostgreSQL
DB_HOST = "localhost"  # Adresse de PostgreSQL (localhost si en local)
DB_NAME = "model_predictions"  # Nom de la base

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 🔹 Initialisation de la base de données
db = SQLAlchemy(app)

# 🔹 Définition de la table pour stocker les prédictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    predicted_value = db.Column(db.Float, nullable=False)

# 🔹 Charger le modèle
MODEL_PATH = "rf_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"⚠ Le modèle {MODEL_PATH} est introuvable.")

# 🔹 Charger les données de test
TEST_DATA_PATH = "clean_test.csv"
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError("⚠ Le fichier clean_test.csv est introuvable !")

test_df = pd.read_csv(TEST_DATA_PATH)
test_df.insert(0, "Select", False)  # Ajouter une colonne de sélection

@app.route('/')
def home():
    return render_template("index.html", data=test_df.to_dict(orient="records"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_data = request.json.get("selected_rows", [])
        df = pd.DataFrame(selected_data)

        # Vérifier et afficher les noms des colonnes
        expected_columns = model.feature_names_in_
        print("Colonnes attendues :", list(expected_columns))
        print("Colonnes reçues :", list(df.columns))

        # Vérification stricte des colonnes
        if not all(col in df.columns for col in expected_columns):
            return jsonify({'error': '⚠ Les noms des colonnes ne correspondent pas !'}), 400
        
        # Réordonner les colonnes avant de faire la prédiction
        df = df[expected_columns]

        # Faire la prédiction
        predictions = model.predict(df)

        return jsonify({'status': 'success', 'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
