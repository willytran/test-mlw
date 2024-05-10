import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from dataProcessing import process_data 

def compute_tsne(featuresData, n_components=3, perplexity=100):
    featuresData.replace([np.inf, -np.inf], np.nan, inplace=True)
    featuresData.dropna(inplace=True)

    customers_clusters = pd.DataFrame()
    customers_clusters['AccountNumber'] = featuresData['AccountNumber']
    features_clean = featuresData.drop(columns='AccountNumber')

    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features_clean))

    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    tsne_features = tsne.fit_transform(features_scaled)

    customers_clusters['tsne_features1'] = tsne_features[:, 0]
    customers_clusters['tsne_features2'] = tsne_features[:, 1]
    customers_clusters['tsne_features3'] = tsne_features[:, 2]

    return customers_clusters

if __name__ == '__main__':
    featuresData = process_data()
    if not featuresData.empty:
        tsne_data = compute_tsne(featuresData)
        print(tsne_data.head(10))
    else:
        print("No data found")
