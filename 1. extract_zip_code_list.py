#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import folium
import pgeocode


# In[2]:


def load_data(path, sep, names):
    l = [pd.read_csv(filename, sep=sep, names=names) for filename in glob.glob(path)]
    return(pd.concat(l, axis=0))


# In[6]:


def add_longi_lati(df, nomi):
    longi = []
    lati = []
    place = []
    for index, row in df.iterrows():
        print('.', end='')
        for i in range(0,100):
            temp_post_code = str(int(str(row['POSTAL_CODE'])[:3])*100 + i)
            query = nomi.query_postal_code(temp_post_code)
            if pd.isna(query['longitude']):
                pass
            else:
                longi.append(query['longitude'])
                lati.append(query['latitude'])
                place.append(query['place_name'])
                break
    df['LONGITUDE'] = longi
    df['LATITUDE'] = lati
    df['PLACE'] = place
    return df


# In[7]:


def main():
    print("Starting.")
    
    names = ['CREDIT_SCORE ', 'FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER_FLAG', 'MATURITY_DATE', 
             'METROPOLITAN_STATISTICAL_AREA', 'MORTGAGE_INSURANCE_PERCENTAGE', 'NUMBER_OF_UNITS', 'OCCUPANCY_STATUS',
             'ORIGINAL_COMBINED_LOAN-TO-VALUE', 'ORIGINAL_DEBT_TO_INCOME_RATIO', 'ORIGINAL_UPB', 'ORIGINAL_LOAN-TO-VALUE',
             'ORIGINAL_INTEREST_RATE', 'CHANNEL', 'PREPAYMENT_PENALTY_MORTGAGE_(PPM)_FLAG', 'PRODUCT_TYPE', 'PROPERTY_STATE',
             'PROPERTY_TYPE', 'POSTAL_CODE', 'LOAN_SEQUENCE_NUMBER', 'LOAN_PURPOSE', 'ORIGINAL_LOAN_TERM',
             'NUMBER_OF_BORROWERS', 'SELLER_NAME', 'SERVICER_NAME', 'SUPER_CONFORMING_FLAG', 'Pre_HARP_LOAN_SEQUENCE_NUMBER']
    path = 'C:\\Users\\Abhilash\\Desktop\\scikit-learn\\SampleInputFiles\\sample_orig_*.txt'
    sep='|'
    
    nomi = pgeocode.Nominatim('us')
    
    df = load_data(path,sep,names)
    post_code_series = df['POSTAL_CODE'].value_counts()

    df_post_code = pd.DataFrame()
    df_post_code['POSTAL_CODE'] = post_code_series.index
    df_post_code['COUNT'] = post_code_series.values
    
    df_longi_lati = add_longi_lati(df_post_code, nomi)
    
    df_longi_lati.to_csv('post_code_longi_lati.csv', index=False)
    
    print("Process Completed.")


# In[8]:


if __name__ == '__main__':

    main()


# In[ ]:




