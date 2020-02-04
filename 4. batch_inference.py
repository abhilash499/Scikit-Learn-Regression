#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import glob
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import seaborn as sns


# In[13]:


def load_data(path, sep, names):
    l = [pd.read_csv(filename, sep=sep, names=names) for filename in glob.glob(path)]
    return(pd.concat(l, axis=0))


# In[78]:


def select_valid_data(df):
    
    # Drop Invalid categorical attributes.
    # Get names of indexes for which 'OCCUPANCY_STATUS' has value 9
    index_os = df[ df['OCCUPANCY_STATUS'] == '9' ].index
    # Delete these row indexes from dataFrame
    df.drop(index_os , inplace=True)

    # Get names of indexes for which 'CHANNEL' has value T or 9
    index_ch = df[ df['CHANNEL'] == 'T'].index + df[ df['CHANNEL'] == '9'].index
    # Delete these row indexes from dataFrame
    df.drop(index_ch , inplace=True)

    # Get names of indexes for which 'PROPERTY_TYPE' has value 99
    index_pt = df[ df['PROPERTY_TYPE'] == '99' ].index
    # Delete these row indexes from dataFrame
    df.drop(index_pt , inplace=True)

    # Get names of indexes for which 'LOAN_PURPOSE' has value R or 9
    index_lp = df[ df['LOAN_PURPOSE'] == 'R' ].index + df[ df['LOAN_PURPOSE'] == '9' ].index
    # Delete these row indexes from dataFrame
    df.drop(index_lp , inplace=True)
    
    # Drop Invalid numerical attributes.
    # Get names of indexes for which 'CREDIT_SCORE' has value 9999
    index_cs = df[ df['CREDIT_SCORE'] == 9999 ].index
    # Delete these row indexes from dataFrame
    df.drop(index_cs , inplace=True)

    # Get names of indexes for which 'MORTGAGE_INSURANCE_PERCENTAGE' has value 999
    index_mip = df[ df['MORTGAGE_INSURANCE_PERCENTAGE'] == 999 ].index
    # Delete these row indexes from dataFrame
    df.drop(index_mip , inplace=True)

    # Get names of indexes for which 'NUMBER_OF_UNITS' has value 99
    index_units = df[ df['NUMBER_OF_UNITS'] == 99 ].index
    # Delete these row indexes from dataFrame
    df.drop(index_units , inplace=True)

    # Get names of indexes for which 'ORIGINAL_COMBINED_LOAN-TO-VALUE' has value 999
    index_ocltv = df[ df['ORIGINAL_COMBINED_LOAN-TO-VALUE'] == 999 ].index
    # Delete these row indexes from dataFrame
    df.drop(index_ocltv , inplace=True)

    # Get names of indexes for which 'ORIGINAL_DEBT_TO_INCOME_RATIO' has value 999
    index_odir = df[ df['ORIGINAL_DEBT_TO_INCOME_RATIO'] == 999 ].index
    # Delete these row indexes from dataFrame
    df.drop(index_odir , inplace=True)

    # Get names of indexes for which 'ORIGINAL_LOAN-TO-VALUE' has value 999
    index_oltv = df[ df['ORIGINAL_LOAN-TO-VALUE'] == 999 ].index
    # Delete these row indexes from dataFrame
    df.drop(index_oltv , inplace=True)

    # Get names of indexes for which 'NUMBER_OF_BORROWERS' has value 99
    index_nob = df[ df['NUMBER_OF_BORROWERS'] == 99 ].index
    # Delete these row indexes from dataFrame
    df.drop(index_nob , inplace=True)
    
    # Get names of indexes for which 'SELLER_NAME' has NEW VALUES.
    index_sn = (df[ df['SELLER_NAME'] == 'SUNTRUSTBANK'].index) 
    df.drop(index_sn , inplace=True)
    
    # Get names of indexes for which 'SELLER_NAME' has NEW VALUES.
    index_sn = (df[ df['SELLER_NAME'] == 'CITIZENSBANK,NATL'].index) 
    df.drop(index_sn , inplace=True)
    
    # Get names of indexes for which 'SELLER_NAME' has NEW VALUES.
    index_sn = (df[ df['SELLER_NAME'] == 'TEXASCAPITALBANK,NA'].index) 
    df.drop(index_sn , inplace=True)
    
     # Get names of indexes for which 'SERVICER_NAME' has NEW VALUES.
    index_sn = (df[ df['SERVICER_NAME'] == 'LOANDEPOTCOM,LLC'].index) 
    df.drop(index_sn , inplace=True)
    
    # Get names of indexes for which 'SERVICER_NAME' has NEW VALUES.
    index_sn = (df[ df['SERVICER_NAME'] == 'FRANKLINAMERICANMTGE'].index) 
    df.drop(index_sn , inplace=True)
    
    # Get names of indexes for which 'SERVICER_NAME' has NEW VALUES.
    index_sn = (df[ df['SERVICER_NAME'] == 'UNITEDSHOREFINANCIAL'].index) 
    df.drop(index_sn , inplace=True)
    
    # Drop Columns 'SUPER_CONFORMING_FLAG' and 'Pre_HARP_LOAN_SEQUENCE_NUMBER'
    df.drop(columns=['MATURITY_DATE', 'PRODUCT_TYPE', 'LOAN_SEQUENCE_NUMBER', 'SUPER_CONFORMING_FLAG', 
                     'Pre_HARP_LOAN_SEQUENCE_NUMBER'], inplace=True)
    
    return df


# In[79]:


names = ['CREDIT_SCORE', 'FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER_FLAG', 'MATURITY_DATE', 'METROPOLITAN_STATISTICAL_AREA', 
 'MORTGAGE_INSURANCE_PERCENTAGE', 'NUMBER_OF_UNITS', 'OCCUPANCY_STATUS', 'ORIGINAL_COMBINED_LOAN-TO-VALUE', 
 'ORIGINAL_DEBT_TO_INCOME_RATIO', 'ORIGINAL_UPB', 'ORIGINAL_LOAN-TO-VALUE', 'ORIGINAL_INTEREST_RATE', 
 'CHANNEL', 'PREPAYMENT_PENALTY_MORTGAGE_(PPM)_FLAG', 'PRODUCT_TYPE', 'PROPERTY_STATE', 'PROPERTY_TYPE', 'POSTAL_CODE', 
 'LOAN_SEQUENCE_NUMBER', 'LOAN_PURPOSE', 'ORIGINAL_LOAN_TERM', 'NUMBER_OF_BORROWERS', 'SELLER_NAME', 'SERVICER_NAME', 
 'SUPER_CONFORMING_FLAG', 'Pre_HARP_LOAN_SEQUENCE_NUMBER']

path = 'C:\\Users\\Abhilash\\Desktop\\Scikit-Learn\\SampleInputFiles\\sample_orig_2018.txt'
sep='|'


# In[80]:


df = load_data(path,sep,names)
df.shape


# In[81]:


df = select_valid_data(df)
df.shape


# In[82]:


X = df.drop('ORIGINAL_INTEREST_RATE', axis=1) # drop labels for training set
y = df['ORIGINAL_INTEREST_RATE']


# In[83]:


class Impute_Attributes(BaseEstimator, TransformerMixin):
    def __init__(self, impute_date=True, impute_ppm=True, impute_msa=True, impute_postal=True):
        self.impute_date = impute_date
        self.impute_ppm = impute_ppm
        self.impute_msa = impute_msa
        self.impute_postal = impute_postal
        
    def fit(self, Z, y=None):
        return self # Do Nothing.
    
    def transform(self, Z, y=None):
        
        X = Z.values
        
        if self.impute_date:
            X[:,dt_ix] = [x%100 for x in X[:,dt_ix]]
        
        if self.impute_ppm:
            X[:,ppm_ix] = ['Y' if x is not 'N' else 'N' for x in X[:,ppm_ix]]
            
        if self.impute_msa:
            X[:, msa_ix] = [1 if x >= 0 else 0 for x in X[:, msa_ix]]
            
        if self.impute_postal:
            X[:, postal_ix] = [postal_cluster_dict[x]['LABEL'] if x in postal_cluster_dict
                               else 10 for x in X[:, postal_ix]]        
        return X


# In[84]:


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]


# In[85]:


cat_cols = ['FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER_FLAG', 'METROPOLITAN_STATISTICAL_AREA', 'OCCUPANCY_STATUS', 'CHANNEL',
            'PREPAYMENT_PENALTY_MORTGAGE_(PPM)_FLAG', 'PROPERTY_STATE', 'PROPERTY_TYPE', 'POSTAL_CODE', 
            'LOAN_PURPOSE', 'SELLER_NAME', 'SERVICER_NAME']


# In[86]:


dt_ix, ppm_ix, msa_ix, postal_ix = [cat_cols.index(col) for col in ['FIRST_PAYMENT_DATE', 
                                                                    'PREPAYMENT_PENALTY_MORTGAGE_(PPM)_FLAG',
                                                                    'METROPOLITAN_STATISTICAL_AREA', 'POSTAL_CODE']]


# In[87]:


my_model_loaded = joblib.load("rate_model.pkl")
postal_cluster_dict = joblib.load("postal_cluster_dict.pkl")


# In[90]:


y_predict = my_model_loaded.predict(X)


# In[96]:


mse = mean_squared_error(y, y_predict)
rmse = np.sqrt(mse)
rmse


# In[99]:


sns.residplot(y, y_predict, color="g")


# In[ ]:




