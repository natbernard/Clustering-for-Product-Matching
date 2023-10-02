import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher, get_close_matches

import warnings
warnings.filterwarnings("ignore")

def category_cleanup(product_list, data):
    product_list = product_list[['Product Name', 'Category', 'Sub category']].applymap(lambda x: str(x).lower().strip())\
                   .drop_duplicates(subset=['Product Name'], keep='first')\
                   .rename(columns={'Product Name': 'product_name'})\
                   .reset_index(drop=True)             
    
    df = data.applymap(lambda x: str(x).lower().strip())\
           .drop_duplicates(subset=['correct_product_match'], keep='first')\
           .rename(columns={'correct_product_match': 'product_name'})\
           .reset_index(drop=True)
           
    df = df.merge(product_list, how='left', on='product_name')
    
    # cleaning categories
    df['Category'] = np.where(df['Category'].isna(), df['category_name'], df['Category'])  
             
    categories = product_list['Category'].unique().tolist()
    
    wrong_categories_df = df[~df['Category'].isin(categories)]
    wrong_categories_df = wrong_categories_df.drop_duplicates(subset='Category', keep='first')

    # cleanup function
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
    df = df.drop(['category_name', 'Category'], axis = 1).\
            rename(columns={'match': 'category_name'})
    
    # cleaning subcategories
    df['Sub category'] = np.where(df['Sub category'].isna(), df['sub_category'], df['Sub category'])
    df = df.drop('sub_category', axis = 1).\
            rename(columns={'Sub category': 'sub_category'})
    
    df = df.drop_duplicates(subset=['product_name'], keep='last').reset_index(drop=True)
    
    print(df.head())
    
    return df


if __name__ == "__main__":
    product_list_df = pd.read_excel('/home/natasha/Documents/Iprocure/Clustering-for-Product-Matching/data/data_v2/product_list.xlsx')   
    category_data = pd.read_csv('/home/natasha/Documents/Iprocure/Clustering-for-Product-Matching/data/data_v2/dirty_category_data.csv')
    
    category_cleanup(product_list_df, category_data)
    
    

    