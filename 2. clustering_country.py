#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import folium
import pgeocode
import requests
from pandas.io.json import json_normalize


# In[10]:


df = pd.read_csv('post_code_longi_lati.csv')


# In[11]:


df.shape


# In[12]:


# Create a dictionary of pincodes as keys and longi/lati as values.
def create_dict(df_longi_lati):
    dict_zip_code={}
    for index, row in df_longi_lati.iterrows():
        zip_code = row['POSTAL_CODE']
        longi = row['LONGITUDE']
        lati = row['LATITUDE']
    
        if zip_code not in dict_zip_code:
            dict_zip_code[zip_code]={
                'LONGITUDE':longi,
                'LATITUDE':lati
            }
    return dict_zip_code


# In[13]:


post_code_dict = create_dict(df)


# In[14]:


# Create a map using dataframe with PLACE, LONGITUDE AND LATITUDE as attributes.
def create_map(df):
    map_temp = folium.Map()
    for lat, lng, place, count in zip(df['LATITUDE'], df['LONGITUDE'], df['PLACE'], df['COUNT']): 
        label = 'Place:{} {} Loan_Count:{}'.format(place, '\n', count)
        label = folium.Popup(label, parse_html=True)
#         folium.Marker([lat, lng]).add_to(map_temp)
#         folium.Circle([lat, lng], radius=4).add_to(map_temp)
        folium.CircleMarker(
            [lat, lng],
            radius=4,
            popup=label,
            color='red',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7,
            parse_html=False).add_to(map_temp)
    return map_temp


# In[15]:


map_us = create_map(df)


# In[16]:


map_us


# In[17]:


# Setup Foursquare Credentials
CLIENT_ID = 'ABCDEFGHIJ123456KLMNOPQRST897362524UVWXYZ' # your Foursquare ID
CLIENT_SECRET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # your Foursquare Secret
VERSION = '20190901' # Foursquare API version

RADIUS = 500 # Radius of search in meters
LIMIT = 300 # Limit the count of search


# In[18]:


df.head()


# In[19]:


def getNearbyVenues(postcode, lati, longi):
    
    venues_list=[]
    
    for postcode, LATI, LONGI in zip(postcode, lati, longi):        
        
        # Setup url for API call
        url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'         .format(CLIENT_ID, CLIENT_SECRET, LATI, LONGI, VERSION, RADIUS, LIMIT)
            
       # API call to get the json file
        results = requests.get(url).json()
        print(results)
        
        # Extract all the necessary categories from the venue.
        venue_details = results['response']['venues']

        category_details = []
        for venue in venue_details:
            category_details.append(venue['categories'])

        for detail in category_details:
            if len(detail) != 0:
                venues_list.append((postcode,detail[0]['name']))


    nearby_venues = pd.DataFrame(venues_list)
    nearby_venues.columns = ['POSTAL_CODE', 'VENUE_CATEGORY']
    
    return(nearby_venues)
    print(pst_code_sum)


# In[20]:


post_codes = df['POSTAL_CODE']
lati = df['LATITUDE']
longi = df['LONGITUDE']
df_us_venues = getNearbyVenues(post_codes, lati, longi)


# In[21]:


df_us_venues.head()


# In[22]:


df_us_venues.shape


# In[23]:


len(np.unique(df_us_venues['VENUE_CATEGORY']))


# In[ ]:


df_us_venues['POSTCODE'].value_counts().max()


# In[ ]:


# Apply One Hot Encoding on df_toronto_venues
venue_onehot = pd.get_dummies(df_us_venues[['VENUE_CATEGORY']],prefix='',prefix_sep='')

print('Shape of venue_onehot: {}'.format(venue_onehot.shape))
print('Total no of Categories in venue_onehot: {}'.format(len(venue_onehot.columns)))
venue_onehot.head()


# In[ ]:


# Add post_code to venue_onehot and move it to 1st column for easy understanding
venue_onehot['POSTCODE'] = df_us_venues['POSTCODE']
new_col_seq = [venue_onehot.columns[-1]] + list(venue_onehot.columns[:-1])
venue_onehot = venue_onehot[new_col_seq]


print('Shape of venue_onehot: {}'.format(venue_onehot.shape))
print('Total no of Categories in venue_onehot: {}'.format(len(venue_onehot.columns)-1))
venue_onehot.head()


# In[ ]:


# Group_by venue_onehot based on neighborhood to match df for joining. Here sum is considered.
neighborhood_onehot = venue_onehot.groupby('POSTCODE').sum().reset_index()

print('Total neighborhoods = Total Postcodes:{}'.format(neighborhood_onehot.shape[0]))
print('Total no of diff venue categories used: {}'.format(neighborhood_onehot.shape[1]-1))
neighborhood_onehot.head()


# In[ ]:


new= list(neighborhood_onehot['POSTCODE'])
main=list(df['POSTAL_CODE'])


# In[ ]:


for x in main:
    if x in new:
        pass
    else:
        print(x)


# In[ ]:


neighborhood_onehot.to_csv('neighborhood.csv',index=False)


# In[ ]:


# Create an Empty dataframe which shows top n venue categories in a neighborhood.
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['POSTAL_CODE']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhood_top_venues = pd.DataFrame(columns=columns)
neighborhood_top_venues['POSTAL_CODE'] = neighborhood_onehot['POSTCODE']

neighborhood_top_venues.head()


# In[ ]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)    
    return row_categories_sorted.index.values[0:num_top_venues]

for row in np.arange(neighborhood_onehot.shape[0]):
    neighborhood_top_venues.iloc[row, 1:] = return_most_common_venues(neighborhood_onehot.iloc[row, :], num_top_venues)
    
neighborhood_top_venues


# In[ ]:


# Apply Kmeans clustering
from sklearn.cluster import KMeans
neighborhood_model_data = neighborhood_onehot.drop(['POSTCODE'], axis=1)

k = 10
kmeans = KMeans(n_clusters=k, random_state=0).fit(neighborhood_model_data)
len(kmeans.labels_)


# In[276]:


# Is done to avoid rerun of complete code for any issues.
neighborhood_top_clustered_venues = neighborhood_top_venues.copy(deep=True)


# In[277]:


# Add labels to neighborhood_top_clustered_venues
neighborhood_top_clustered_venues.insert(1, 'Cluster Labels', kmeans.labels_)
neighborhood_top_clustered_venues


# In[278]:


# Join df_data and neighborhood_top_clustered_venues on Neighborhood for final o/p dataset.
clustered_us = pd.merge(df, neighborhood_top_clustered_venues, on='POSTAL_CODE', how='inner')


# In[279]:


clustered_us


# In[283]:


clustered_us.shape


# In[305]:


map_us_clustered = folium.Map()

latitude = clustered_us['LATITUDE']
longitude = clustered_us['LONGITUDE']
postcode = clustered_us['POSTAL_CODE']
clusters = clustered_us['Cluster Labels']

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'gray', 'cadetblue', 'darkblue']
# add markers to map
for lat, lng, pstcde, cluster in zip(latitude, longitude, postcode, clusters):
    label = ' {} {}, {}, {}'.format(lat, lng, pstcde, cluster)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=4,
        popup=label,
        color=colors[cluster-1],
        fill=True,
        fill_color=colors[cluster-1],
        fill_opacity=0.7,
        parse_html=False).add_to(map_us_clustered)


# In[306]:


map_us_clustered


# In[307]:


clustered_us.to_csv('clustered_us.csv')


# In[ ]:




