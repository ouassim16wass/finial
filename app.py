from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Configuration PostgreSQL
DB_USER = "myuser"
DB_PASSWORD = "mypassword"
DB_HOST = "localhost"
DB_NAME = "model_predictions"

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Définition de la table pour stocker les prédictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    predicted_value = db.Column(db.Float, nullable=False)

# Charger le modèle
MODEL_PATH = "rf_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Le modèle {MODEL_PATH} est introuvable.")

# Charger les données et sélectionner les colonnes utiles
TEST_DATA_PATH = "clean_test.csv"
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError("Le fichier clean_test.csv est introuvable !")

test_df = pd.read_csv(TEST_DATA_PATH)
test_df = test_df[['Id', 'LotArea', 'YearBuilt', 'OverallQual', 'GrLivArea']]

test_df.insert(0, "Select", False)

@app.route('/')
def home():
    return render_template("index.html", data=test_df.to_dict(orient="records"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_data = request.json.get("selected_rows", [])
        df = pd.DataFrame(selected_data)
        expected_columns = model.feature_names_in_

        if not all(col in df.columns for col in expected_columns):
            return jsonify({'error': 'Colonnes incorrectes !'}), 400

        df = df[expected_columns]
        predictions = model.predict(df)

        results = [{'Id': row['Id'], 'SalePrice': pred} for row, pred in zip(selected_data, predictions)]
        return jsonify({'status': 'success', 'results': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
