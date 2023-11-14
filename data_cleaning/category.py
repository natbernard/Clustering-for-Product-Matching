import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher, get_close_matches

import warnings
warnings.filterwarnings("ignore")

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



if __name__ == "__main__":
    iprocure_product_df = pd.read_excel('../data/data_v2/product_list.xlsx')   
    data = pd.read_csv('../clean_prod_datasets/ipos_products.csv')
    
    category_cleanup(data, iprocure_product_df)
    
    

    