from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder

# Load the saved model
loaded_model = joblib.load('logistic_regression_model.joblib')
print("Model loaded")

# Load the unique values from the file
cat_cols_unique_values_loaded = joblib.load('cat_cols_unique_values.joblib')
print("Unique categorical features loaded")

# Load the data
transactionData = pd.read_csv('../Feb2023.csv')
X = transactionData.drop(columns=['AmlProcessingStatus'])
y = transactionData['AmlProcessingStatus'].apply(lambda x: 0 if x == "UNSUSPICIOUS" else 1)
print("Data loaded")

# Load the encoder objects
categorical_cols = ['AcceptanceMethod', 'SourceCurrency', 'Direction', 'Type', 'AmlTransactionStatus', 'CounterpartyCountry']
encoders = {}
for col in categorical_cols:
    encoders[col] = joblib.load(f'{col}_encoder.joblib')
print("Encoders loaded")

# Encode categorical variables
for col in categorical_cols:
    try:
        X[col] = encoders[col].transform(X[col])
    except KeyError:
        # handle unknown labels in categorical columns
        X[col] = X[col].apply(lambda x: -1 if x not in encoders[col].classes_ else x)
        X[col] = encoders[col].transform(X[col])

X['Amount'] = (X['Amount'] - X['Amount'].min())/(X['Amount'].max() - X['Amount'].min())
X['Balance'] = (X['Balance'] - X['Balance'].min())/(X['Balance'].max() - X['Balance'].min())

print("Data transformed")

Xnumpy = pd.get_dummies(X).to_numpy()

y_pred = loaded_model.predict(Xnumpy)

acc = accuracy_score(y, y_pred)
pre = precision_score(y, y_pred)
rec = recall_score(y, y_pred)

print("Results on February (Acc - Pre - Rec) :", acc, pre, rec)