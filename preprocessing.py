# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA

# Charger les fichiers
train_file_path = "train.csv"
test_file_path = "test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Supprimer 'Id' car inutile pour l'entraînement
train_df.drop(columns=['Id'], inplace=True, errors='ignore')
test_ids = test_df['Id']
test_df.drop(columns=['Id'], inplace=True, errors='ignore')

# Séparer les colonnes numériques et catégoriques
num_features_train = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice', errors='ignore')
num_features_test = test_df.select_dtypes(include=['int64', 'float64']).columns

cat_features_train = train_df.select_dtypes(include=['object']).columns
cat_features_test = test_df.select_dtypes(include=['object']).columns

# Remplacement des valeurs manquantes
train_df[num_features_train] = train_df[num_features_train].fillna(train_df[num_features_train].median())
test_df[num_features_test] = test_df[num_features_test].fillna(test_df[num_features_test].median())

for col in cat_features_train:
    mode_value = train_df[col].mode()[0] if not train_df[col].mode().empty else "Unknown"
    train_df[col] = train_df[col].fillna(mode_value)
    test_df[col] = test_df[col].fillna(mode_value)

# Encodage des variables catégoriques
label_encoders = {}
for col in cat_features_train:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Suppression des variables fortement corrélées
correlation_matrix = train_df.corr()
variables_to_remove = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.90:
            variables_to_remove.add(correlation_matrix.columns[j])

train_df_reduced = train_df.drop(columns=variables_to_remove)

# Application de PCA
num_data = train_df_reduced.select_dtypes(include=['int64', 'float64']).drop(columns=['SalePrice'], errors='ignore')
pca = PCA(n_components=0.95)
pca_transformed = pca.fit_transform(num_data)

# Sauvegarde des fichiers prétraités
train_df_reduced.to_csv("clean_train_reduced.csv", index=False)
test_df.to_csv("clean_test.csv", index=False)
