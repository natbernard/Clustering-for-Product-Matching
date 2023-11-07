import pandas as pd
import numpy as np
from difflib import SequenceMatcher, get_close_matches

import re

def manufacturer_clustering(data, master_df):
    # slicing for manufacturer name, match, and match score
    df = data[['manufacturer_name', 'best_manufacturer_match', 'manufacturer_match_score']].\
                    applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
                    
    df_manufacturer = df[df['manufacturer_match_score'] < 0.8]
    
    df_non_dup = df_manufacturer.drop_duplicates(subset='manufacturer_name', keep='first').reset_index(drop=True)
    df_non_dup['word_count'] = df_non_dup['manufacturer_name'].apply(lambda x: len(x.split()) if isinstance(x, str) else 1)
    df_non_dup['manufacturer_name'] = df_non_dup['manufacturer_name'].astype('str')
    
    manufacturer_list = []

    for index, row in df_non_dup.iterrows():
        word_count = row['word_count']
        manufacturer_name = row['manufacturer_name']
        
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
            compare.update({i: get_close_matches(i, df_manufacturer_slice['manufacturer_slice'].to_list(), 20, 0.85)})
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
    
    similar_strings = []

    for i, row in df_non_dup.iterrows():
        string = row['manufacturer_name']
        value = row['match']
        
        # Check if any other row has a similar value in 'col2'
        similar_rows = df_non_dup[df_non_dup['match'].apply(lambda x: x == value)]
        
        # Extract the strings from 'col1' in similar rows
        similar_strings.append(similar_rows['manufacturer_name'].tolist())

    df_similar_strings = pd.DataFrame({'similar_strings': similar_strings})
    df_similar_strings = df_similar_strings['similar_strings'].drop_duplicates(keep='first').to_frame().reset_index(drop=True)
    
    df_unique_match = df_non_dup['match'].\
                            drop_duplicates(keep='first').\
                            to_frame().\
                            reset_index(drop=True)
                            
    df_unique_match = pd.concat([df_unique_match, df_similar_strings],axis = 1)
    df_unique_match['average_length'] = df_unique_match['similar_strings'].\
                                                    apply(lambda x: round(sum(len(word.split()) for word in x) / len(x), 0))
    df_unique_match['similar_strings'] = df_unique_match['similar_strings'].apply(lambda x: ' '.join(x))
    
    # extracting most common words from each cluster in order
    cluster_word_freq = {}

    for id, row in df_unique_match.iterrows():
        cluster = row['similar_strings']
        
        words = re.split(r'\s+|-|\(|\)|/|\\|\||,', cluster)
        for word in words:
            if id in cluster_word_freq:
                cluster_word_freq[id][word] = cluster_word_freq[id].get(word, 0) + 1
            else:
                cluster_word_freq[id] = {word: 1}
        
    for id in cluster_word_freq:
        cluster_word_freq[id] = sorted(cluster_word_freq[id].items(), key=lambda x: x[1], reverse=True)

    cluster_word_freq_df = pd.DataFrame.from_dict(cluster_word_freq.items())
    cluster_word_freq_df.rename(columns={0: 'id', 1: 'word_freq'}, inplace=True)
    
    cluster_word_freq_df = pd.concat([cluster_word_freq_df, df_unique_match[['average_length']]], axis=1)

    for i, row in cluster_word_freq_df.iterrows():
        lst = row['word_freq']
        number = int(row['average_length'])
        
        cluster_name = ' '.join(word[0] for word in lst[:number])
        cluster_word_freq_df.at[i, 'cluster_name'] = cluster_name
        
    
    # cleaning against master list
    cluster_word_freq_df['cluster_name'] = cluster_word_freq_df['cluster_name'].astype('str')
    
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

    found_df = cluster_word_freq_df['cluster_name'].apply(lambda x: get_closest_match(x, master_list))
    found_df = found_df.apply(pd.Series)

    cluster_word_freq_df = pd.concat([cluster_word_freq_df, found_df], axis=1)
    cluster_word_freq_df['cluster_name'] = np.where(cluster_word_freq_df['best_score'] >= 0.8, cluster_word_freq_df['best_match'], cluster_word_freq_df['cluster_name'])
    
    df_unique_match = pd.concat([df_unique_match, cluster_word_freq_df], axis=1)
    df_unique_match['match'] = df_unique_match['match'].apply(lambda x: ' '.join(x))
    df_non_dup['match'] = df_non_dup['match'].apply(lambda x: ' '.join(x))
    df_non_dup = df_non_dup.merge(df_unique_match, how='left', on='match')

    # picking between original best match and new cluster name 
    def compare(row):
        comparison = {}
        i = row['manufacturer_name']
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

    columns_to_convert = ['manufacturer_name', 'cluster_name', 'best_manufacturer_match']
    df_non_dup[columns_to_convert] = df_non_dup[columns_to_convert].applymap(convert_to_string)
    
    df_non_dup[['final_match', 'score']] = df_non_dup.apply(lambda row: compare(row), axis=1)
    df_non_dup['final_match'] = df_non_dup['final_match'].apply(lambda x: ' '.join(x))
    df_non_dup['score'] = df_non_dup['score'].apply(lambda x: x[0])    
    df_non_dup['correct_manufacturer_match'] = np.where(df_non_dup['score'] >= 0.8, df_non_dup['final_match'], df_non_dup['cluster_name'])
    
    # data['product_manufacturer'] = data['product_manufacturer'].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
    # data = data.merge(df_non_dup[['product_manufacturer', 'correct_manufacturer_match']], how='left', on='product_manufacturer')
    # data['correct_manufacturer_match'] = np.where(data['correct_manufacturer_match'].isna(), data['best_manufacturer_match'], data['correct_manufacturer_match'])
    
    print(df_non_dup.head())
    
    # return data
    

if __name__ == "__main__":
    data =  pd.read_csv('../data/data_v2/subsequent_manufacturers.csv')
    data = data[:100]
    master_df = pd.read_csv('../data/data_v1/master_list.csv')
    manufacturer_clustering(data, master_df)