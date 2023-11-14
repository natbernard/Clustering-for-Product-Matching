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
from pyspark.sql import SparkSession
from pyspark import StorageLevel

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(
    parent_dir
)

# from data_cleaning.product import matching_by_clustering, internal_string_matching, master_string_matching
# from data_cleaning.manufacturer import manufacturer_clustering
# from data_cleaning.category import category_cleanup
# from data_cleaning.type import type_cleanup

import warnings
warnings.filterwarnings('ignore')


# def read_from_bqtable(bquery):
#     client = bigquery.Client()
#     bq_data = client.query(bquery).to_dataframe()
#     return bq_data

def write_table_to_bigquery(mode, data, dataset, table, bucket):
        data.write. \
        format("bigquery"). \
        mode(mode). \
        option("checkpointLocation", "gs://{0}/{1}".format(bucket, "restore-point")). \
        option("temporaryGcsBucket", bucket). \
        save("{0}.{1}".format(dataset, table))


# master_df = master_df = pd.read_csv('../data/data_v1/master_list.csv')
# iprocure_product_df = pd.read_excel('../data/data_v2/product_list.xlsx')   
# original_data = pd.read_csv('../ipos_products.csv')


# STEP 1: Product names
def matching_by_clustering(data):
    df = data[['product_name', 'best_product_match', 'product_match_score']].\
                applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)
    
    # getting records with match score less than 80%
    to_cluster = df[df['product_match_score'] < 0.8]
    unique_names = to_cluster['product_name'].dropna().unique()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(unique_names)
    
    # # clustering product names
    # def optimal_clusters(matrix):
    #     clusters = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    #     silhouette_scores = []
        
    #     for cluster in clusters:
    #         kmeans = KMeans(n_clusters=cluster,random_state=42)
    #         kmeans.fit(matrix)
    #         cluster_labels = kmeans.labels_
            
    #         silhouette_avg = silhouette_score(matrix, cluster_labels)
    #         silhouette_scores.append(silhouette_avg)
            
    #     optimal_num_clusters = clusters[silhouette_scores.index(max(silhouette_scores))]
        
    #     return optimal_num_clusters
    
    # cluster_size = optimal_clusters(tfidf_matrix)
    
    kmeans = KMeans(n_clusters=10000, random_state=42).fit(tfidf_matrix)
    labels = kmeans.labels_
    
    # creating a dataframe with the clusters
    cluster_to_name = {}
    for label in set(labels):
        indices = np.where(labels == label)[0]
        names = unique_names[indices].tolist()
        cluster_to_name[label] = names
        
    cluster_df = pd.DataFrame.from_dict(cluster_to_name.items())
    cluster_df.rename(columns={0: 'cluster_id', 1: 'product_names'}, inplace=True)
    
    unique_names_df = pd.DataFrame({'product_name': unique_names,
                                    'label': labels})
    
    # creating a cluster name    
    cluster_df['cluster_name'] = cluster_df['product_names'].apply(lambda x: x[0])
    cluster_df = cluster_df.rename(columns={'cluster_id': 'label'})
    
    # merging cluster names with original product names
    merge_df = to_cluster.merge(unique_names_df, how='left', on='product_name')
    merge_df = merge_df.merge(cluster_df[['label', 'cluster_name']], how='left', on='label')
        
    return merge_df
    
    






# STEP 2: Product names
def internal_string_matching(clustered_data):
    unique_clustered_data = clustered_data[['product_name', 'cluster_name', 'best_product_match']].\
                            drop_duplicates(subset=['product_name'], keep='first').reset_index(drop=True)
    
    # picking between original best match and new cluster name 
    def compare(row):
        comparison = {}
        i = row['product_name']
        prods_list = row[['cluster_name', 'best_product_match']].tolist()
        if isinstance(i, str):
            comparison.update({i: get_close_matches(i, prods_list, n=1, cutoff=0.1)})
        product_name = list(comparison.keys()) if comparison else None
        match = []
        score = []
        if comparison:
            for key, value in comparison.items():
                if value:
                    match.append(value[0])
                    score.append(round(SequenceMatcher(None, i, value[0]).ratio(), 2))
                else:
                    match.append(None)
                    score.append(None)
        else:
            match.append(None)
            score.append(None)
                
        return pd.Series([match, score], index = ['match', 'score'])
    
    unique_clustered_data[['match', 'score']] = [compare(row) for _, row in unique_clustered_data.iterrows()]
    unique_clustered_data[['match', 'score']] = unique_clustered_data[['match', 'score']].apply(lambda x: x[0])
    unique_clustered_data['go_to_match'] = np.where(unique_clustered_data['score'] >= 0.8, unique_clustered_data['match'], unique_clustered_data['cluster_name'])
    
    return unique_clustered_data








# STEP 3: Product names
def master_string_matching(products_df, master_df, unique_clustered_data):
    master_list = master_df['product_name'].to_list()
    matches_cache = {}

    # cleaning cluster names against master list
    def get_closest_match(word, possibilities: list[str]):
        word = str(word).lower()
        if found := matches_cache.get(word):
            return found

        matches = get_close_matches(word, possibilities, n=1, cutoff=0.0)
        match = matches[0] if matches else ''
        score = round(SequenceMatcher(None, word, match).ratio(), 2)
        found = {'best_match': match, 'best_score': score}
        matches_cache[word] = found

        return found
    
    matched_df = unique_clustered_data['go_to_match'].apply(lambda x: get_closest_match(x, master_list))
    matched_df = matched_df.apply(pd.Series)
    
    unique_clustered_data = pd.concat([unique_clustered_data, matched_df], axis=1)
    unique_clustered_data['correct_product_match'] = np.where(unique_clustered_data['best_score'] >= 0.8, unique_clustered_data['best_match'], unique_clustered_data['go_to_match'])
    
    products_df['product_name'] = products_df['product_name'].apply(lambda x: x.lower().strip() if isinstance(x, str) else x)
    products_df = products_df.merge(unique_clustered_data[['product_name', 'correct_product_match']], how='left', on='product_name')
    products_df['correct_product_match'] = np.where(products_df['correct_product_match'].isna(), products_df['best_product_match'], products_df['correct_product_match'])
           
    return products_df







# STEP 4: Manufacturer names
def manufacturer_clustering(data, master_df):
    df = data[['product_manufacturer', 'best_manufacturer_match', 'manufacturer_match_score']].\
                    applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
        
    df_manufacturer = df[df['manufacturer_match_score'] < 0.8]
        
    df_non_dup = df_manufacturer.drop_duplicates(subset='product_manufacturer', keep='first').reset_index(drop=True)
    df_non_dup['word_count'] = df_non_dup['product_manufacturer'].apply(lambda x: len(x.split()) if isinstance(x, str) else 1)
    df_non_dup['product_manufacturer'] = df_non_dup['product_manufacturer'].astype('str')
    
    # slicing manufacturer names
    manufacturer_list = []

    for index, row in df_non_dup.iterrows():
        word_count = row['word_count']
        manufacturer_name = row['product_manufacturer']
        
        if word_count in [1,2]:
            manufacturer_slice = manufacturer_name.strip().split()[:1]
            manufacturer_list.append(' '.join(manufacturer_slice))
        elif word_count in [3,4,5]:
            manufacturer_slice = manufacturer_name.strip().split()[:2]
            manufacturer_list.append(' '.join(manufacturer_slice))
        else:
            manufacturer_slice = manufacturer_name.strip().split()[:3]
            manufacturer_list.append(' '.join(manufacturer_slice))
            
    df_non_dup['manufacturer_slice'] = manufacturer_list  
    
    df_manufacturer_slice = df_non_dup['manufacturer_slice'].\
                                        drop_duplicates(keep='first').\
                                        to_frame().\
                                        reset_index(drop=True)
    
    # grouping similar manufacturer names
    matched = []
    def compare(i):
        compare = {}
        if i in matched:
            compare.update({i: ''})
        else:
            compare.update({i: get_close_matches(i, df_manufacturer_slice['manufacturer_slice'].to_list(), 20, 0.80)})
        matched.extend([item for sublist in compare.values() for item in sublist])
        manufacturer_slice = list(compare.keys())
        match = []
        for key, items in compare.items():
            match.append(items)
        return pd.Series([manufacturer_slice, match],index=['manufacturer_slice', 'match'])

    cleaned_manufacturers_df = pd.DataFrame()
    cleaned_manufacturers_df[['manufacturer_slice', 'match']] = df_manufacturer_slice['manufacturer_slice'].apply(lambda x: compare(x))
    cleaned_manufacturers_df = cleaned_manufacturers_df.applymap(lambda x: x[0] if x else '')
    
    cleaned_manufacturers_df['manufacturer_slice'] = cleaned_manufacturers_df['manufacturer_slice'].astype('str')
    
    for i, row in cleaned_manufacturers_df.iterrows():
        string = row['manufacturer_slice']
        lst = row['match']
        
        if not lst:
            for prev_i in range(i):
                prev_lst = cleaned_manufacturers_df.at[prev_i, 'match']
                if string in prev_lst:
                    cleaned_manufacturers_df.at[i, 'match'] = prev_lst
                    break  
    
    df_non_dup = pd.merge(df_non_dup, cleaned_manufacturers_df, how='left', on='manufacturer_slice')
    
    # getting full manufacturer names
    similar_strings = []

    for i, row in df_non_dup.iterrows():
        string = row['product_manufacturer']
        value = row['match']
        
        similar_rows = df_non_dup[df_non_dup['match'].apply(lambda x: x == value)]
        
        similar_strings.append(similar_rows['product_manufacturer'].tolist())

    df_similar_strings = pd.DataFrame({'similar_strings': similar_strings})
    df_similar_strings = df_similar_strings['similar_strings'].drop_duplicates(keep='first').to_frame().reset_index(drop=True)
    
    df_unique_match = df_non_dup['match'].\
                            drop_duplicates(keep='first').\
                            to_frame().\
                            reset_index(drop=True)
                            
    df_unique_match = pd.concat([df_unique_match, df_similar_strings],axis = 1)
    
    # creating cluster names
    df_unique_match['cluster_name'] = df_unique_match['similar_strings'].apply(lambda x: x[0])        
    
    # cleaning against master list
    df_unique_match['cluster_name'] = df_unique_match['cluster_name'].astype('str')
    
    master_df['manufacturer_name'] = master_df['manufacturer_name'].astype('str')    

    matches_cache = {}
    master_list = master_df['manufacturer_name'].to_list()

    def get_closest_match(word, possibilities: list[str]):
        word = str(word).lower()
        if found := matches_cache.get(word):
            return found

        matches = get_close_matches(word, possibilities, n=1, cutoff=0.0)
        match = matches[0] if matches else ''
        score = round(SequenceMatcher(None, word, match).ratio(), 2)
        found = {'best_match': match, 'best_score': score}
        matches_cache[word] = found

        return found

    found_df = df_unique_match['cluster_name'].apply(lambda x: get_closest_match(x, master_list))
    found_df = found_df.apply(pd.Series)

    df_unique_match = pd.concat([df_unique_match, found_df], axis=1)
    df_unique_match['cluster_name'] = np.where(df_unique_match['best_score'] >= 0.8, df_unique_match['best_match'], df_unique_match['cluster_name'])
    
    df_unique_match['match'] = df_unique_match['match'].apply(lambda x: ' '.join(x))
    df_non_dup['match'] = df_non_dup['match'].apply(lambda x: ' '.join(x))
    df_non_dup = df_non_dup.merge(df_unique_match, how='left', on='match')

    # picking between original best match and new cluster name 
    def compare(row):
        comparison = {}
        i = row['product_manufacturer']
        prods_list = row[['cluster_name', 'best_manufacturer_match']].tolist()
        if isinstance(i, str):
            comparison.update({i: get_close_matches(i, prods_list, n=1, cutoff=0.1)})
        product_name = list(comparison.keys()) if comparison else None
        match = []
        score = []
        if comparison:
            for key, value in comparison.items():
                if value:
                    match.append(value[0])
                    score.append(round(SequenceMatcher(None, i, value[0]).ratio(), 2))
                else:
                    match.append(None)
                    score.append(None)
        else:
            match.append(None)
            score.append(None)
                    
        return pd.Series([match, score], index = ['match', 'score'])
    
    def convert_to_string(value):
        return str(value)

    columns_to_convert = ['product_manufacturer', 'cluster_name', 'best_manufacturer_match']
    df_non_dup[columns_to_convert] = df_non_dup[columns_to_convert].applymap(convert_to_string)
    
    df_non_dup[['final_match', 'score']] = df_non_dup.apply(lambda row: compare(row), axis=1)
    df_non_dup['final_match'] = df_non_dup['final_match'].apply(lambda x: ' '.join(x))
    df_non_dup['score'] = df_non_dup['score'].apply(lambda x: x[0])    
    df_non_dup['correct_manufacturer_match'] = np.where(df_non_dup['score'] >= 0.8, df_non_dup['final_match'], df_non_dup['cluster_name'])

    data['product_manufacturer'] = data['product_manufacturer'].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
    data = data.merge(df_non_dup[['product_manufacturer', 'correct_manufacturer_match']], how='left', on='product_manufacturer')
    data['correct_manufacturer_match'] = np.where(data['correct_manufacturer_match'].isna(), data['best_manufacturer_match'], data['correct_manufacturer_match'])
        
    return data








# STEP 5: Categories
def category_cleanup(data, iprocure_product_df):
    product_list = iprocure_product_df[['Product Name', 'Category', 'Sub category']].applymap(lambda x: str(x).lower().strip())\
                    .drop_duplicates(subset=['Product Name'], keep='first')\
                    .rename(columns={'Product Name': 'product_name'})\
                    .reset_index(drop=True)             
    
    categories = product_list['Category'].unique().tolist()

    df = data[['correct_product_match', 'product_category', 'sub_category']].applymap(lambda x: str(x).lower().strip())\
                    .rename(columns={'correct_product_match': 'product_name'})\
                    .reset_index(drop=True)
           
    category_mask = df['product_category'].isin(categories)
    no_category_mask = ~df['product_name'].isin(df.loc[category_mask, 'product_name'])
    keep_rows_mask = category_mask | no_category_mask
    
    df = df[keep_rows_mask].drop_duplicates(subset=['product_name'], keep='first')

    df = df.merge(product_list, how='left', on='product_name')
    
    # cleaning categories
    df['Category'] = np.where(df['Category'].isna(), df['product_category'], df['Category'])  
                 
    wrong_categories_df = df[~df['Category'].isin(categories)]
    wrong_categories_df = wrong_categories_df.drop_duplicates(subset='Category', keep='first')

    # cleaning against iprocure categories
    def compare(i):
        comparison = {}
        if isinstance(i, str):
            comparison.update({i: get_close_matches(i, categories, n=1, cutoff=0.1)})
        category = list(comparison.keys()) if comparison else None
        match = []
        score = []
        if comparison:
            for key, value in comparison.items():
                if value:
                    match.append(value[0])
                    score.append(round(SequenceMatcher(None, i, value[0]).ratio(), 2))
                else:
                    match.append(None)
                    score.append(None)
        else:
            match.append(None)
            score.append(None)
                
        return pd.Series([category, match, score], index = ['category', 'match', 'score'])

    cleaned_categories_df = pd.DataFrame()
    cleaned_categories_df[['category', 'match', 'score']] = wrong_categories_df['Category'].apply(lambda x: compare(x))
    cleaned_categories_df = cleaned_categories_df.applymap(lambda x: x[0] if x else '')
    
    category_matches_df = cleaned_categories_df[cleaned_categories_df['score'] >= 0.7]
    category_matches_df = category_matches_df.rename(columns={'category': 'Category'})
    
    df = df.merge(category_matches_df[['Category', 'match']], how='left', on='Category')
    df['match'] = np.where(df['match'].isna(), df['Category'], df['match'])
    df = df.drop('Category', axis = 1).\
            rename(columns={'match': 'correct_category_name'})
    
    # cleaning subcategories
    df['Sub category'] = np.where(df['Sub category'].isna(), df['sub_category'], df['Sub category'])
    df = df.drop('sub_category', axis = 1).\
            rename(columns={'Sub category': 'correct_sub_category',
                            'product_name': 'correct_product_match'})
    
    df = df.drop_duplicates(subset=['correct_product_match'], keep='first').reset_index(drop=True)
    
    data['correct_product_match'] = data['correct_product_match'].apply(lambda x: str(x).lower().strip())
    data = data.merge(df[['correct_product_match', 'correct_category_name', 'correct_sub_category']], how='left', on='correct_product_match')

    return data









# STEP 5: Product types
def type_cleanup(data, iprocure_product_df):
    product_list_df = iprocure_product_df[['Product Name', 'Type']].\
                                    applymap(lambda x: str(x).lower().strip()).\
                                    drop_duplicates(subset=['Product Name', 'Type'], keep='first').\
                                    reset_index(drop=True)
                                    
    df = data[['correct_product_match', 'product_type']].\
                                    applymap(lambda x: str(x).lower().strip()).\
                                    drop_duplicates(subset=['correct_product_match', 'product_type'], keep='first').\
                                    reset_index(drop=True)
                                    
    # extracting number and unit parts
    def extract_parts(value):
        pattern = r'(\d+(?:\.\d+)?|½|¼|¾|\d+\/\d+)(\s*[a-zA-Z]+)'
        matches = re.match(pattern, value)

        if matches:
            digits_part = matches.group(1)
            letters_part = matches.group(2)
            return digits_part, letters_part.strip()
        else:
            return value, value
        
    # applying the function to the column and creating new columns
    product_list_df[['number', 'unit']] = product_list_df['Type'].apply(lambda x: pd.Series(extract_parts(str(x))))
    df[['number', 'unit']] = df['product_type'].apply(lambda x: pd.Series(extract_parts(str(x))))
    
    units = product_list_df['unit'].unique().tolist()
    units_df = df['unit'].drop_duplicates(keep='first').reset_index(drop=True).to_frame()

    # cleanup function to clean units
    def compare(i):
        comparison = {}
        if isinstance(i, str):
            comparison.update({i: get_close_matches(i, units, n=1, cutoff=0.1)})
        unit = list(comparison.keys()) if comparison else None
        match = []
        score = []
        if comparison:
            for key, value in comparison.items():
                if value:
                    match.append(value[0])
                    score.append(round(SequenceMatcher(None, i, value[0]).ratio(), 2))
                else:
                    match.append(None)
                    score.append(None)
        else:
            match.append(None)
            score.append(None)

        return pd.Series([unit, match, score], index = ['unit', 'match', 'score'])

    cleaned_types_df = pd.DataFrame()
    cleaned_types_df[['unit', 'match', 'score']] = units_df['unit'].apply(lambda x: compare(x))
    cleaned_types_df = cleaned_types_df.applymap(lambda x: x[0] if x else '')

    # clustering unmatched units
    to_cluster_df = cleaned_types_df[cleaned_types_df['score'] < 0.7].reset_index(drop=True)
    
    matched = []
    def compare(i):
        compare = {}
        if i in matched:
            compare.update({i: ''})
        else:
            compare.update({i: get_close_matches(i, to_cluster_df['unit'].to_list(), 20, 0.7)})
        matched.extend([item for sublist in compare.values() for item in sublist])
        unit = list(compare.keys())
        match = []
        for key, items in compare.items():
            match.append(items)
        return pd.Series([unit, match], index=['unit', 'match'])


    cluster_cleaned_df = pd.DataFrame()
    cluster_cleaned_df[['unit', 'match']] = to_cluster_df['unit'].apply(lambda x: compare(x))
    cluster_cleaned_df = cluster_cleaned_df.applymap(lambda x: x[0] if x else '')

    cluster_cleaned_df['unit'] = cluster_cleaned_df['unit'].astype('str')
    
    for i, row in cluster_cleaned_df.iterrows():
        string = row['unit']
        lst = row['match']

        if not lst:
            for prev_i in range(i):
                prev_lst = cluster_cleaned_df.at[prev_i, 'match']
                if string in prev_lst:
                    cluster_cleaned_df.at[i, 'match'] = prev_lst
                    break  
                
    # creating a cluster name    
    cluster_cleaned_df['cluster_name'] = cluster_cleaned_df['match'].apply(lambda x: x[0])

    clean_types = cleaned_types_df[cleaned_types_df['score'] >= 0.7]
    df = df.merge(clean_types[['unit', 'match']], how='left', on='unit')
    df = df.merge(cluster_cleaned_df[['unit', 'cluster_name']], how='left', on='unit')
    df['match'] = np.where(df['match'].isna(), df['cluster_name'], df['match'])
    df['match'] = np.where(df['match'].isna(), df['unit'], df['match'])

    def add_correct_product_type_column(df):
        def is_numeric(value):
            pattern = r'^\d+(\.\d+)?|½|¼|¾|\d+\/\d+$'
            return bool(re.match(pattern, value))

        df['correct_product_type'] = df.apply(lambda row: row['number'] + row['match'] if row['number'] != row['match'] and is_numeric(row['number']) and row['match'].isalpha() else row['match'], axis=1)

    df[['number', 'match']] = df[['number', 'match']].astype(str)
    add_correct_product_type_column(df)
    
    df = df[['correct_product_match', 'product_type', 'correct_product_type']].\
                drop_duplicates(subset=['correct_product_match', 'product_type', 'correct_product_type'], keep='first')
                
    data[['correct_product_match', 'product_type']] = data[['correct_product_match', 'product_type']].applymap(lambda x: str(x).lower().strip())
    data = data.merge(df[['correct_product_match', 'product_type', 'correct_product_type']], how='left', on=['correct_product_match', 'product_type'])

    print(data.head())

    return data





service_account_json = '../bigquery_credentials/credentials.json'
tmp_bucket = 'iprocure-edw'
dataset_name = 'iprocure-edw.iprocure_edw'
table_name = 'products_cleaned_2.0'
project_id = 'iprocure-edw'
table_id = 'iprocure-edw.iprocure_edw.products_cleaned'
query = f"""
        SELECT *
        FROM {table_id}
        """
        
master_df = pd.read_csv(f'gs://{tmp_bucket}/data-cleaning/master_list.csv', encoding='utf-8', na_values=['NA', 'N/A'])
iprocure_product_df = pd.read_csv(f'gs://{tmp_bucket}/data-cleaning/iprocure_products.csv', encoding='utf-8', na_values=['NA', 'N/A'])
original_data = pd.read_csv(f'gs://{tmp_bucket}/data-cleaning/ipos_products.csv', encoding='utf-8', na_values=['NA', 'N/A'])
print('Finished loading data!')

clustered_data = matching_by_clustering(original_data)
print('Finished first step!')
print(clustered_data.head())

unique_clustered_data = internal_string_matching(clustered_data)
print('Finished step 2!')
print(unique_clustered_data.head())

df = master_string_matching(original_data, master_df, unique_clustered_data)
print('Finished step 3!')
print(df.head())

df = manufacturer_clustering(df, master_df)
print('Finished step 4!')
print(df.head())

df = category_cleanup(df, iprocure_product_df)
print('Finished step 5!')
print(df.head())

df = type_cleanup(df, iprocure_product_df)
print('Finished last step!')
print(df.head())

df.to_csv(f'gs://{tmp_bucket}/data-cleaning/new_cleaned_products.csv', index=False)

print('Done!')
# spark = SparkSession.\
#                 builder.\
#                 appName("pandas-to-spark").\
#                 getOrCreate()

# spark_df = spark.createDataFrame(df)
# spark_df.persist(StorageLevel.MEMORY_ONLY_SER)

# write_table_to_bigquery(mode="append",
#                         data=df,
#                         dataset=dataset_name,
#                         table=table_name,
#                         bucket=tmp_bucket)

# spark_df.unpersist()