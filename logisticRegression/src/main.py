import logisticRegression as lr
import pandas as pd
import numpy as np

from category_encoders import TargetEncoder
encoder = TargetEncoder()
slice = 100000

# Extraction of the data and corresponding labels
transactionData = pd.read_csv('../Jan2023.csv')
transactionData = transactionData.head(slice)
y = np.zeros(slice)
for i in range(slice):
    if transactionData['AmlProcessingStatus'][i] == "UNSUSPICIOUS": y[i] = 0
    else: y[i] = 1
transactionData = transactionData.drop(columns=['AmlProcessingStatus'])
# print(transactionData)
# print(y)

# Classifiers categories : AcceptanceMethod, SourceCurrency, Direction, Type, AmlTransactionStatus, CounterpartyCountry
transactionData['AcceptanceMethod'] = encoder.fit_transform(transactionData['AcceptanceMethod'], y)
transactionData['SourceCurrency'] = encoder.fit_transform(transactionData['SourceCurrency'], y)
transactionData['Direction'] = encoder.fit_transform(transactionData['Direction'], y)
transactionData['Type'] = encoder.fit_transform(transactionData['Type'], y)
transactionData['AmlTransactionStatus'] = encoder.fit_transform(transactionData['AmlTransactionStatus'], y)
transactionData['CounterpartyCountry'] = encoder.fit_transform(transactionData['CounterpartyCountry'], y)

# Standardisation of numerical values
transactionData['Amount'] = (transactionData['Amount'] - transactionData['Amount'].min())/(transactionData['Amount'].max() - transactionData['Amount'].min())
transactionData['Balance'] = (transactionData['Balance'] - transactionData['Balance'].min())/(transactionData['Balance'].max() - transactionData['Balance'].min())

# print(transactionData)

numpyTransactionData = pd.get_dummies(transactionData).to_numpy()

# beta = np.zeros(len(numpyTransactionData[0]))

with open("../meanBeta.txt") as f:
    beta = np.array([float(line.strip()) for line in f])

# print(lr.countOnes(y))

metrics, meanBeta = lr.kFoldCrossValidation(numpyTransactionData, y, beta, 0.001, 100, 5)
# print(metrics)
# print(meanBeta)

meanBeta = map(str, meanBeta.tolist())
with open("../meanBeta.txt", 'w') as file:
    file.write('\n'.join(meanBeta))