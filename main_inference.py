# Data Process
import pandas as pd
import numpy as np 

# Data visulization
import plotly.express as px
import plotly
import matplotlib.pyplot as plt

# IO
from pathlib import Path
import pickle

# Feature & Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# custom function 
from src.data import data_preprocess


# read csv data file
home = str(Path.home())
data = pd.read_csv(home + '/Startup-Analysis/data/raw/startup_data.csv')

# only include necessary columns
drop_col = ['Unnamed: 0', 'Unnamed: 6', 'state_code.1', 'object_id']
data = data_preprocess.drop_unnecessary_col(data, drop_col)

# convert negative value to 0
feat_need_clean = ['age_first_funding_year',
                  'age_last_funding_year',
                  'age_first_milestone_year',
                  'age_last_milestone_year']
data = data_preprocess.neg_to_zero(data, feat_need_clean)

# impute missing data using regression
missing_val = ['age_first_milestone_year', 'age_last_milestone_year']
independent_col = ['age_last_funding_year']
data = data_preprocess.impute_using_reg(data, missing_val, independent_col)

# derive decile for features
col_to_process = ['latitude', 'longitude']
data = data_preprocess.derive_decile(data, col_to_process)

# load model artifact
artifact_name = "xgb_tuned_model.pkl"
xgb_model_loaded = pickle.load(open(home  + '/Startup-Analysis/models/model_artifact/' + artifact_name, "rb"))

# load feature list
feat_file_name = 'feature.pkl'
feat = pickle.load(open(home  + '/Startup-Analysis/models/feat/' + feat_file_name, "rb"))

# score the input dataset
X = data[feat]
y_pred = xgb_model_loaded.predict_proba(X)

data['prob'] = y_pred[:,1]
data[feat + ['prob']].to_csv(home + '/Startup-Analysis/data/processed/results.csv')
