
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.svm import SVC

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
def print_cols(df):
    for col in df.columns:
        print(col)

def convert_cols_to_floats(df):
    
    for col in df.columns:
        df[col]= df[col].astype(float)
    return df


def convert_death_cancer_col(df):
    death_from_dict = {
    'Living':0
    ,'Died of Other Causes':0
    ,'Died of Disease':1
    }
    df.death_from_cancer.replace(death_from_dict, inplace =True)
    df.death_from_cancer.fillna(0, inplace = True)
    return df


rna_df = pd.read_csv('../data/capstone2.mrn_df2_laptop.csv')
full_df = pd.read_csv('../data/METABRIC_RNA_Mutation.csv')

rna_df.info(verbose = True)
# rna_df['death_from_cancer'] = full_df.death_from_cancer
# df.death_from_cancer.isna().sum()
# rna_df = convert_death_cancer_col(rna_df)
# rna_df = convert_cols_to_floats(rna_df)
# x = convert_death_cancer_col(rna_df)
# rna_df.to_csv(r'../data/capstone2.mrn_df2_laptop.csv')

# print(x.death_from_cancer.isnull().sum())