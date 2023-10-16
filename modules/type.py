import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher, get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


def type_cleanup(data, iprocure_product_df):
    product_list_df = iprocure_product_df[['Product Name', 'Type']].\
                                    applymap(lambda x: str(x).lower().strip()).\
                                    drop_duplicates(subset=['Product Name', 'Type'], keep='first').\
                                    reset_index(drop=True)
                                    
    df = data[['correct_product_match',	'product_type']].\
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
        
    # applying the function to the column and create new columns
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

        # checking if the string exists in any list of previous rows
        if not lst:
            for prev_i in range(i):
                prev_lst = cluster_cleaned_df.at[prev_i, 'match']
                if string in prev_lst:
                    cluster_cleaned_df.at[i, 'match'] = prev_lst
                    break  
                
    cluster_cleaned_df['match_concat'] = cluster_cleaned_df['match'].apply(lambda x:' '.join(x))
    cluster_cleaned_df['match_split'] = cluster_cleaned_df['match_concat'].apply(lambda x: re.split(r'\s+|-|\(|\)|/|\\|\||,', x))

    # extracting most common words from each cluster in order
    cluster_word_freq = {}

    for id, row in cluster_cleaned_df.iterrows():
        cluster = row['match_concat']

        words = re.split(r'\s+|-|\(|\)|/|\\|\||,', cluster)
        words = [word for word in words if word.strip()]

        for word in words:
            if id in cluster_word_freq:
                cluster_word_freq[id][word] = cluster_word_freq[id].get(word, 0) + 1
            else:
                cluster_word_freq[id] = {word: 1}

    for id in cluster_word_freq:
        cluster_word_freq[id] = sorted(cluster_word_freq[id].items(), key=lambda x: x[1], reverse=True)

    cluster_word_freq_df = pd.DataFrame.from_dict(cluster_word_freq.items())
    cluster_word_freq_df.rename(columns={0: 'id', 1: 'word_freq'}, inplace=True)
    
    cluster_word_freq_df['cluster_name'] = cluster_word_freq_df['word_freq'].apply(lambda x: ''.join(word[0] for word in x[:1]))
    
    cluster_names = cluster_word_freq_df['cluster_name'].to_list()

    def find_cluster_name(string):
        for i in cluster_names:
            if i in string:
                return i

    cluster_cleaned_df['cluster_name'] = cluster_cleaned_df['match_split'].apply(find_cluster_name)
    
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
                drop_duplicates(subset=['correct_product_match', 'correct_product_type'], keep='first')
                
    data[['correct_product_match', 'product_type']] = data[['correct_product_match', 'product_type']].applymap(lambda x: str(x).lower().strip())
    data = data.merge(df[['correct_product_match', 'product_type', 'correct_product_type']], how='left', on=['correct_product_match', 'product_type'])

    print(data.head(20))
    
    return df

def main():
    iprocure_product_df = pd.read_excel('/home/natasha/Documents/Iprocure/Clustering-for-Product-Matching/data/data_v2/product_list.xlsx')
    data = pd.read_csv('/home/natasha/Documents/Iprocure/Clustering-for-Product-Matching/data/data_v2/dirty_product_types.csv')
    
    type_cleanup(data, iprocure_product_df)
    
if __name__=="__main__":
    main()


    
    

    
        
    
                                    
    
                                    
    
    
    
    