import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime

# Config
META_URL = 'https://green-got.metabaseapp.com'
API_KEY = 'mb_6nGgD4/4QnKRrEvUQ8Zq0/LX+b1PX8fR06nFHGiuX4g='
QUESTION_ID = 2407

HEADERS = {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY
}

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(question_id, headers, url):
    question_url = f'{url}/api/card/{question_id}/query/json'

    try:
        response = requests.post(question_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except requests.RequestException as e:
        logging.error(f"Error fetching Metabase data: {e}")
        return pd.DataFrame()  
    
def process_data():
    rawData = fetch_data(QUESTION_ID, HEADERS, META_URL)

    if rawData.empty:
        logging.info("No data fetched. Exit")
        return pd.DataFrame() 
    
    try :
        rawData = rawData.fillna(0)

        # Convert numeric columns to appropriate data types
        numeric_columns = ['Nb SEPAOUT', 'Nb transactions', 'Nb CREDIT', 'Nb DEBIT', 'Nb WITHDRAWAL', 'SponsoringCount',
                        'Volume CREDIT', 'Ecart-type Volume transactions', 'Arrondi', 'Nb interaction app',
                        'Volume DEBIT', 'Moyenne CREDIT', 'Moyenne DEBIT', 'Nb hits']
        for col in numeric_columns:
            rawData[col] = pd.to_numeric(rawData[col], errors='coerce').fillna(0)

        rawData['AccountCreatedAt'] = pd.to_datetime(rawData['AccountCreatedAt'], format='%d.%m.%Y, %H:%M',
                                                    dayfirst=True).dt.tz_localize('Europe/Paris')

        featuresData = pd.DataFrame()
        featuresData['AccountNumber'] = rawData['AccountNumber']
        featuresData['nb_withdrawal_ratio'] = rawData['Nb WITHDRAWAL'] / rawData['Nb transactions']
        featuresData['nb_sepaout_ratio'] = rawData['Nb SEPAOUT'] / rawData['Nb transactions']
        featuresData['avg_credit_volume'] = rawData['Volume CREDIT'] / 100 / rawData['Nb CREDIT']
        featuresData['avg_debit_volume'] = rawData['Volume DEBIT'] / 100 / rawData['Nb DEBIT']
        featuresData['debit_credit_ratio'] = rawData['Nb DEBIT'] / rawData['Nb CREDIT']
        featuresData['credit_transactions_ratio'] = rawData['Nb CREDIT'] / rawData['Nb transactions']
        featuresData['debit_transactions_ratio'] = rawData['Nb DEBIT'] / rawData['Nb transactions']
        featuresData['volume_transaction_std'] = rawData['Ecart-type Volume transactions'] / 100
        featuresData['hits_ratio'] = rawData['Nb hits'] / rawData['Nb transactions']

        today = datetime.today()
        num_days = (today - rawData['AccountCreatedAt'].dt.tz_convert(None)).dt.days
        featuresData['age'] = num_days
        featuresData['transactions_per_day'] = rawData['Nb transactions'] / num_days
        featuresData['interaction_ratio'] = rawData['Nb interaction app'] / num_days
        featuresData['rounding_total'] = rawData['Arrondi'] / 100
        featuresData['sponsored'] = rawData['SponsoredBy'].apply(lambda x: 1 if x != 'SKIPPED' else 0)
        featuresData['sponsorships'] = rawData['SponsoringCount']

        featuresData = featuresData.fillna(0)

        return featuresData
    
    except Exception as e:
        logging.error(f"Error during data processing: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    features = process_data()
    if not features.empty:
        print(features.head(10))
    else:
        logging.info("No data found")