import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from difflib import SequenceMatcher, get_close_matches
from google.cloud import bigquery
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(
    parent_dir
)

from data_cleaning.product import matching_by_clustering, internal_string_matching, master_string_matching
from data_cleaning.manufacturer import manufacturer_clustering
from data_cleaning.category import category_cleanup
from data_cleaning.type import type_cleanup




import warnings
warnings.filterwarnings('ignore')

service_account_json = '/home/natasha/Documents/Iprocure/Clustering-for-Product-Matching/bigquery_credentials/credentials.json'
tmp_bucket = 'iprocure-edw'
project_id = 'iprocure-edw'
table_id = 'iprocure-edw.iprocure_edw.products_cleaned'
query = f"""
        SELECT *
        FROM {table_id}
        """

def read_from_bqtable(bquery):
    client = bigquery.Client.from_service_account_json(service_account_json)
    bq_data = client.query(bquery).to_dataframe()
    return bq_data

print('Finished reading from BQ!')

original_data = read_from_bqtable(query)
master_df = f'gs://{tmp_bucket}/data-cleaning/master_list.csv'
iprocure_product_df = f'gs://{tmp_bucket}/data-cleaning/iprocure_products.csv'

clustered_data = matching_by_clustering(original_data)
print('Finished step 1!')

unique_clustered_data = internal_string_matching(clustered_data)
print('Finished step 2!')

df = master_string_matching(original_data, master_df, unique_clustered_data)
print('Finished step 3!')

df = manufacturer_clustering(df, master_df)
print('Finished step 4!')

df = category_cleanup(df, iprocure_product_df)
print('Finished step 5!')

df = type_cleanup(df, iprocure_product_df)
print('Finished step 6!')

df.to_csv('cleaned_products.csv', index=False)

print(df.head())










