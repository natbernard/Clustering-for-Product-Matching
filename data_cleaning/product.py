import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from difflib import SequenceMatcher, get_close_matches
import re

import warnings
warnings.filterwarnings('ignore')


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


    
if __name__ == "__main__":    
    master_df = master_df = pd.read_csv('../data/data_v1/master_list.csv')
    iprocure_product_df = pd.read_excel('../data/data_v2/product_list.xlsx')   
    data = pd.read_csv('../ipos_products.csv')
    
    clustered_data = matching_by_clustering(data)
    unique_clustered_data = internal_string_matching(clustered_data)
    master_string_matching(data, master_df, unique_clustered_data)
    


    
    

    
    
    