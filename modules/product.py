import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from difflib import SequenceMatcher, get_close_matches
import re

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)

vectorizer = TfidfVectorizer()


def matching_by_clustering(data):
    df = data[['product_name', 'best_product_match', 'product_match_score']].\
                applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

    to_cluster = df[df['product_match_score'] < 0.8]
    unique_names = to_cluster['product_name'].unique()
    
    tfidf_matrix = vectorizer.fit_transform(unique_names)
    
    def optimal_clusters(matrix):
        clusters = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        silhouette_scores = []
        
        for cluster in clusters:
            kmeans = KMeans(n_clusters=cluster,random_state=42)
            kmeans.fit(matrix)
            cluster_labels = kmeans.labels_
            
            silhouette_avg = silhouette_score(matrix, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
        optimal_num_clusters = clusters[silhouette_scores.index(max(silhouette_scores))]
        
        return optimal_num_clusters
    
    cluster_size = optimal_clusters(tfidf_matrix)
    
    kmeans = KMeans(n_clusters=cluster_size, random_state=42).fit(tfidf_matrix)
    labels = kmeans.labels_
    
    # creating a dataframe of the clusters
    cluster_to_name = {}
    for label in set(labels):
        indices = np.where(labels == label)[0]
        names = unique_names[indices].tolist()
        cluster_to_name[label] = names
        
    # viewing the clusters
    cluster_df = pd.DataFrame.from_dict(cluster_to_name.items())
    cluster_df.rename(columns={0: 'cluster_id', 1: 'product_names'}, inplace=True)
    cluster_df.set_index('cluster_id', inplace=True)
    
    unique_names_df = pd.DataFrame({'product_name': unique_names,
                                    'label': labels})
    
    # extracting most common words from each cluster in order
    cluster_word_freq = {}

    for doc, cluster_label in zip(unique_names, labels):
        words = re.split(r'\s+|-|\(|\)|/|\\|\||,', doc)
        for word in words:
            if cluster_label in cluster_word_freq:
                cluster_word_freq[cluster_label][word] = cluster_word_freq[cluster_label].get(word, 0) + 1
            else:
                cluster_word_freq[cluster_label] = {word: 1}
        
    for cluster_label in cluster_word_freq:
        cluster_word_freq[cluster_label] = sorted(cluster_word_freq[cluster_label].items(), key=lambda x: x[1], reverse=True)
        
    cluster_word_freq_df = pd.DataFrame.from_dict(cluster_word_freq.items())
    cluster_word_freq_df.rename(columns={0: 'label', 1: 'word_freq'}, inplace=True)

    cluster_word_freq_df['cluster_name'] = cluster_word_freq_df['word_freq'].apply(lambda x: ' '.join(word[0] for word in x[:3] if word[0] != ' '))
    
    # merging cluster names with original product names
    merge_df = to_cluster.merge(unique_names_df, how='left', on='product_name')
    merge_df = merge_df.merge(cluster_word_freq_df[['label', 'cluster_name']], how='left', on='label')
    
    print(merge_df.head())
    
    return merge_df
    
    
    
def internal_string_matching(clustered_data):
    unique_clustered_data = data[['product_name', 'cluster_name', 'best_product_match']].\
                            drop_duplicates(subset=['product_name'], keep='first').reset_index(drop=True)
    
    # cleanup function
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
    unique_clustered_data['go_to_match'] = np.where(unique_clustered_data['score'] >= 0.65, unique_clustered_data['match'], unique_clustered_data['cluster_name'])

    print(unique_clustered_data.head())
    
    return unique_clustered_data



def master_string_matching(products_df, master_list, unique_clustered_data):
    master_list = master_list['product_name'].to_list()
    matches_cache = {}

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
    
    print(products_df.head())
    
    return products_df
    
    
    

if __name__ == "__main__":
    master_list = pd.read_csv('/home/natasha/Documents/Iprocure/Clustering-for-Product-Matching/data/data_v1/master_list.csv')
    data = pd.read_csv('/home/natasha/Documents/Iprocure/Clustering-for-Product-Matching/data/data_v2/subsequent_unmatched_products.csv')
    
    clustered_data = matching_by_clustering(data)
    unique_clustered_data = internal_string_matching(clustered_data)
    master_string_matching(data, master_list, unique_clustered_data)
    
    matching_by_clustering(data)



    
    

    
    
    