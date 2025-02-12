from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Configuration de la base de données PostgreSQL (via Docker Compose)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://myuser:mypassword@localhost/model_predictions")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
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
    raise FileNotFoundError(f"⚠️ Le modèle {MODEL_PATH} est introuvable.")

@app.route('/')
def home():
    return jsonify({'message': 'Bienvenue sur l’API de prédiction !'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_data = request.json.get("selected_rows", [])
        df = pd.DataFrame(selected_data)

        expected_columns = model.feature_names_in_
        if not all(col in df.columns for col in expected_columns):
            return jsonify({'error': '⚠️ Données incorrectes'}), 400

        df = df[expected_columns]

        predictions = model.predict(df)

        for pred in predictions:
            new_prediction = Prediction(predicted_value=float(pred))
            db.session.add(new_prediction)
        db.session.commit()

        return jsonify({'status': 'success', 'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000)
