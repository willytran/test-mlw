import pandas as pd
from clustering import compute_kmeans
from tsne import compute_tsne
from dataProcessing import process_data
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def save_to_csv(df, directory, filename):
    """
    Save the given DataFrame to a CSV file within the specified directory.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        directory (str): The directory where the CSV file should be saved.
        filename (str): The name of the CSV file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it does not exist
        logging.info(f"Created directory: {directory}")

    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")

def main():
    try:
        # Process data, perform t-SNE and clustering
        featuresData = process_data()
        if not featuresData.empty:
            tsne_data = compute_tsne(featuresData)

            #Add date to filename
            current_date = datetime.now().strftime('%d_%m_%Y')  # Day_Month_Year format
            clustering_results = compute_kmeans(tsne_data)
            filename = f'clustering_results_{current_date}.csv'


            # Save the clustering results to a CSV file
            save_to_csv(clustering_results, 'clustering/csv',filename)
        else:
            logging.warning("No data found to process.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
