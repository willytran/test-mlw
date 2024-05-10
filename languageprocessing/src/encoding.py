import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from imblearn.combine import SMOTETomek
from fuzzywuzzy import fuzz
import joblib

# Load the data
transactionDataAug = pd.read_csv('../Aug2023.csv')
transactionDataSep = pd.read_csv('../Sep2023.csv')
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
transactionDataGlobal = transactionDataGlobal.drop(['CreatedAt','Rounding0Amount', 'Rounding1Amount', 'Rounding2Amount'], axis=1)

# Enregistrez le DataFrame mis à jour dans un fichier si nécessaire
transactionDataGlobal.to_csv('AugSep2023_encoded.csv', index=False)