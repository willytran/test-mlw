import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from imblearn.combine import SMOTETomek
from fuzzywuzzy import fuzz
import joblib

# ENCODING

# Load the data
transactionDataAug = pd.read_csv('../Aug2023.csv')
transactionDataAug = transactionDataAug.dropna(subset=['Balance'])
transactionDataAug = transactionDataAug.head(100000)
transactionDataSep = pd.read_csv('../Sep2023.csv')
transactionDataSep = transactionDataSep.dropna(subset=['Balance'])
transactionDataGlobal = pd.concat([transactionDataAug, transactionDataSep])

# Save the unique values of each categorical column
categorical_cols = ['AcceptanceMethod', 'SourceCurrency', 'Direction', 'Type', 'Status', 'Users → Gender', 'Users → RiskRating']
cat_cols_unique_values = {}
for col in categorical_cols:
    le = LabelEncoder()
    transactionDataGlobal[col] = le.fit_transform(transactionDataGlobal[col])
    cat_cols_unique_values[col] = list(le.classes_)

    # Save the encoder object
    joblib.dump(le, 'encoding/'f'{col}_encoder.joblib')

# Save the unique values using joblib
joblib.dump(cat_cols_unique_values, 'encoding/cat_cols_unique_values.joblib')
print("Global encoding done")

# 1. Calculer la colonne RoudingAmount
transactionDataGlobal['RoudingAmount'] = transactionDataGlobal[['Rounding0Amount', 'Rounding1Amount', 'Rounding2Amount']].sum(axis=1).fillna(0)

# 2. Calculer la colonne fuzzRatio
transactionDataGlobal['fuzzRatio'] = transactionDataGlobal.apply(lambda row: fuzz.partial_ratio(row['Nom complet'], row['Counterparty']), axis=1)

# 3. Calculer la colonne NbApparition
nb_apparition = transactionDataGlobal['Counterparty'].value_counts().reset_index()
nb_apparition.columns = ['Counterparty', 'NbApparition']
transactionDataGlobal = transactionDataGlobal.merge(nb_apparition, on='Counterparty', how='left')

# Supprimer les colonnes inutiles
transactionDataGlobal = transactionDataGlobal.drop(['CreatedAt','Rounding0Amount', 'Rounding1Amount', 'Rounding2Amount','Counterparty','Nom complet'], axis=1)
print("New columns computed")

# Enregistrez le DataFrame mis à jour dans un fichier si nécessaire
transactionDataGlobal.to_csv('AugSep2023_encoded.csv', index=False)

# TRAINING

# 1. Calculer la colonne RoudingAmount
transactionDataAug['RoudingAmount'] = transactionDataAug[['Rounding0Amount', 'Rounding1Amount', 'Rounding2Amount']].sum(axis=1).fillna(0)

# 2. Calculer la colonne fuzzRatio
transactionDataAug['fuzzRatio'] = transactionDataAug.apply(lambda row: fuzz.partial_ratio(row['Nom complet'], row['Counterparty']), axis=1)

# 3. Calculer la colonne NbApparition
nb_apparition = transactionDataAug['Counterparty'].value_counts().reset_index()
nb_apparition.columns = ['Counterparty', 'NbApparition']
transactionDataAug = transactionDataAug.merge(nb_apparition, on='Counterparty', how='left')

# Supprimer les colonnes inutiles
transactionDataAug = transactionDataAug.drop(['CreatedAt','Rounding0Amount', 'Rounding1Amount', 'Rounding2Amount','Counterparty','Nom complet'], axis=1)

X = transactionDataAug.drop(columns=['AmlProcessingStatus'])
y = transactionDataAug['AmlProcessingStatus'].apply(lambda x: 0 if x == "UNSUSPICIOUS" else 1)
# Load the encoder objects
categorical_cols = ['AcceptanceMethod', 'SourceCurrency', 'Direction', 'Type', 'Status', 'Users → Gender', 'Users → RiskRating']
encoders = {}
for col in categorical_cols:
    encoders[col] = joblib.load('encoding/'f'{col}_encoder.joblib')

# Encode categorical variables
for col in categorical_cols:
    X[col] = encoders[col].transform(X[col])
print("August data encoded")

# Standardize large numerical values
X['Amount'] = (X['Amount'] - X['Amount'].min()) / (X['Amount'].max() - X['Amount'].min())
X['Balance'] = (X['Balance'] - X['Balance'].min()) / (X['Balance'].max() - X['Balance'].min())
X['fuzzRatio'] = (X['fuzzRatio'] - X['fuzzRatio'].min()) / (X['fuzzRatio'].max() - X['fuzzRatio'].min())
X['RoudingAmount'] = (X['RoudingAmount'] - X['RoudingAmount'].min()) / (X['RoudingAmount'].max() - X['RoudingAmount'].min())
X['SourceCurrency'] = (X['SourceCurrency'] - X['SourceCurrency'].min()) / (X['SourceCurrency'].max() - X['SourceCurrency'].min())

X.to_csv('Aug2023_encoded_withoutY.csv', index=False)

# Combine SMOTE and Tomek links to balance the dataset
smt = SMOTETomek(sampling_strategy=0.5, random_state=42)
X, y = smt.fit_resample(X, y)

# Convert to numpy arrays
X = pd.get_dummies(X).to_numpy()
y = y.to_numpy()

# Define logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Define k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
for train, test in kfold.split(X):
    # Fit the model on the training set
    model.fit(X[train], y[train])
    # Predict on the test set
    y_pred = model.predict(X[test])
    # Compute accuracy, precision and recall
    accuracy_scores.append(accuracy_score(y[test], y_pred))
    precision_scores.append(precision_score(y[test], y_pred))
    recall_scores.append(recall_score(y[test], y_pred))

print("Results on August (Acc - Pre - Rec) :")
for i in range(len(accuracy_scores)):
    print(accuracy_scores[i], precision_scores[i], recall_scores[i])

# Save the trained model
joblib.dump(model, 'logistic_regression_model.joblib')