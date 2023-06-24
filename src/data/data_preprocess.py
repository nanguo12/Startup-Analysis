# -*- coding: utf-8 -*-

''' 

This script contains useful functions for pre-process the dataset 
====================================================================
Explaination: 

- drop_unnecessary_col: function that will drop the given columns

- neg_to_zero: function that will convert negative value into 0, used when feature value cannot be negative

- impute_using_reg: function that can impute the missing value of a column based on a linear regression, 
                    use only when there are correlations among those variables 


- derive_decile: function that can derive variable into decile


'''

# Data Process
import pandas as pd
import numpy as np 

# Model
from sklearn.linear_model import LinearRegression


def drop_unnecessary_col(data: pd.DataFrame, col_list: list[str]) -> pd.DataFrame:
    ''' 
    data: input dataset, a pandas dataframe
    col_list: a list of columns that need to drop 
    
    '''
    
    col = [c for c in data.columns if c not in col_list]
    data = data[col]
    
    return data


def neg_to_zero(data: pd.DataFrame , col_list: list[str]) -> pd.DataFrame:
    
    '''
    
    data: input dataset, a pandas dataframe 
    col_list: a list of columns that need to clean 
    
    '''
    for f in col_list:
        data[f] = np.where(data[f] < 0, 0, data[f])
    return data


def impute_using_reg(data: pd.DataFrame, missing_val_col: list[str], independent_col_list) -> pd.DataFrame:

    for f in missing_val_col:
        # exclude NaN value and build a linear regression model 
        data_subset = data[data[f].isna() == False][independent_col_list + [f]]
        
        lr = LinearRegression()
        lr.fit(data_subset[independent_col_list], data_subset[f])
       

        # apply the linear regression model to impute missing data 
        impute_col_name = f + '_impute'
        data[impute_col_name] = np.where(data[f].isna(), 
                                       lr.predict(data[independent_col_list]), 
                                       data[f]
                                      )

    return data

def derive_decile(data: pd.DataFrame, col_to_process: list[str]) -> pd.DataFrame:
    for f in col_to_process:
        new_col_name = f + '_decile'
        data[new_col_name] = pd.qcut(data[f], 10,
                            labels = False)
    return data 

    
    