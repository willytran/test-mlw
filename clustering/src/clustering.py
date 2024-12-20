import pandas as pd
from sklearn.cluster import KMeans
from tsne import compute_tsne  
from dataProcessing import process_data  
import logging
logging.basicConfig(level=logging.INFO)

def compute_kmeans(tsne_data):

    customers_clusters = pd.DataFrame()
    customers_clusters['AccountNumber'] = tsne_data['AccountNumber']
    tsne_clean = tsne_data.drop(columns='AccountNumber')

    kmeans = KMeans(n_clusters=27, random_state=42)
    kmeans_clusters = kmeans.fit_predict(tsne_clean)

    customers_clusters['tsne_features1'] = tsne_clean.iloc[:, 0]
    customers_clusters['tsne_features2'] = tsne_clean.iloc[:, 1]
    customers_clusters['tsne_features3'] = tsne_clean.iloc[:, 2]
    customers_clusters['Cluster'] = kmeans_clusters

    return customers_clusters

def main():
    try:
        featuresData = process_data()
        if not featuresData.empty:
            tsne_data = compute_tsne(featuresData)
            customers_clusters = compute_kmeans(tsne_data)
            logging.info(customers_clusters.head(10))
        else:
            logging.warning("No data found")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()