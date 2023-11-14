import pandas as pd
import numpy as np
from difflib import SequenceMatcher, get_close_matches

import re

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
    


if __name__ == "__main__":
    data = pd.read_csv('../ipos_products.csv')
    master_df = pd.read_csv('../data/data_v1/master_list.csv')
    
    manufacturer_clustering(data, master_df)