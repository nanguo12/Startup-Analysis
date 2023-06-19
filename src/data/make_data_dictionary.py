# -*- coding: utf-8 -*-
import pandas as pd 
class create_data_dictionary:
    
    def __init__(self):
        '''This class provides functions to quickly develop a data dictionary for your data set'''
        return None
    
    def make_my_data_dictionary(self, df):
        '''Create an initial data dictionary excluding definitions for meaning of features '''
        
        col_ = df.columns
        df_DataDict = {}

        for col in col_:
            df_DataDict[col] = {

                'Type': str(df.dtypes[col]),
                'Length': len(df[col]),
                'Null_Count': sum(df[col].isna()),
                'Size(Memory)': df.memory_usage()[col],
                'Definition': str('')
            }

        df_DD = pd.DataFrame(df_DataDict)

        return df_DD
    
    def define_data_meaning(self, df_data_dictionary):
        ''''Quickly provide input regarding each column meaning and transpose into a usable dictionary'''
        col_ = df_data_dictionary.columns
        d = 'Definition'

        for col in col_:
            df_data_dictionary[col][d] = input('Provide a data definition for {}'.format(col))
        
        df_data_dictionary = df_data_dictionary.transpose()

        return df_data_dictionary