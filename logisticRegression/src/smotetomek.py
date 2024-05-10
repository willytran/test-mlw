import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from imblearn.combine import SMOTETomek
import joblib

# from category_encoders import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Load the data
transactionDataJan = pd.read_csv('../Jan2023.csv')
transactionDataFeb = pd.read_csv('../Feb2023.csv')
transactionDataGlobal = pd.concat([transactionDataJan, transactionDataFeb])

# Save the unique values of each categorical column
categorical_cols = ['AcceptanceMethod', 'SourceCurrency', 'Direction', 'Type', 'AmlTransactionStatus', 'CounterpartyCountry']
cat_cols_unique_values = {}
for col in categorical_cols:
    le = LabelEncoder()
    transactionDataGlobal[col] = le.fit_transform(transactionDataGlobal[col])
    cat_cols_unique_values[col] = list(le.classes_)

    # Save the encoder object
    joblib.dump(le, f'{col}_encoder.joblib')

# Save the unique values using joblib
joblib.dump(cat_cols_unique_values, 'cat_cols_unique_values.joblib')
print("Global encoding done")

X = transactionDataJan.drop(columns=['AmlProcessingStatus'])
y = transactionDataJan['AmlProcessingStatus'].apply(lambda x: 0 if x == "UNSUSPICIOUS" else 1)
# Load the encoder objects
categorical_cols = ['AcceptanceMethod', 'SourceCurrency', 'Direction', 'Type', 'AmlTransactionStatus', 'CounterpartyCountry']
encoders = {}
for col in categorical_cols:
    encoders[col] = joblib.load(f'{col}_encoder.joblib')

# Encode categorical variables
for col in categorical_cols:
    X[col] = encoders[col].transform(X[col])
print("January data encoded")

# Standardize numerical values
X['Amount'] = (X['Amount'] - X['Amount'].min()) / (X['Amount'].max() - X['Amount'].min())
X['Balance'] = (X['Balance'] - X['Balance'].min()) / (X['Balance'].max() - X['Balance'].min())

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

print("Results on January (Acc - Pre - Rec) :")
for i in range(len(accuracy_scores)):
    print(accuracy_scores[i], precision_scores[i], recall_scores[i])

# Save the trained model
joblib.dump(model, 'logistic_regression_model.joblib')