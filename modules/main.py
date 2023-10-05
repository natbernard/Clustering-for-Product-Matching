from product import matching_by_clustering, internal_string_matching, master_string_matching
from manufacturer import manufacturer_clustering
from category import category_cleanup
from type import type_cleanup

import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from difflib import SequenceMatcher, get_close_matches
from google.cloud import bigquery

import warnings
warnings.filterwarnings('ignore')


tmp_bucket = 'iprocure-edw'
project_id = 'iprocure-edw'
table_id = 'iprocure-edw.iprocure_edw.full_scoring_data'
query = f"""
        SELECT *
        FROM {table_id}
        """

def read_from_bqtable(bquery):
    client = bigquery.Client()
    bq_data = client.query(query).to_dataframe()
    return bq_data

original_data = read_from_bqtable(query)
master_df = f'gs://{tmp_bucket}/data-cleaning/final_product_list.csv'
iprocure_product_df = f'gs://{tmp_bucket}/data-cleaning/final_product_list.csv'

clustered_data = matching_by_clustering(original_data)
unique_clustered_data = internal_string_matching(clustered_data)
df = master_string_matching(original_data, master_df, unique_clustered_data)
df = manufacturer_clustering(df, master_df)
df = type_cleanup(df, iprocure_product_df)










