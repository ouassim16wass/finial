# predict.py
import joblib
import pandas as pd

# Charger le modèle Random Forest
rf_model = joblib.load('rf_model.pkl')

# Charger les nouvelles données
test_df = pd.read_csv("clean_test.csv")

# Effectuer les prédictions
predictions = rf_model.predict(test_df)

# Sauvegarder les prédictions dans un fichier CSV
output_df = pd.DataFrame({"Id": range(1, len(predictions) + 1), "Predicted SalePrice": predictions})
output_df.to_csv("predictions.csv", index=False)

print("Prédictions sauvegardées dans predictions.csv")
