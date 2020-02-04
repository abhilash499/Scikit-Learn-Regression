#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all libraries
from urllib.request import urlopen
from bs4 import BeautifulSoup
from zipfile import ZipFile
from io import BytesIO
import requests
import os


# In[2]:


def payloadCreation(user, password):
    creds={'username': user,'password': password }
    return creds


# In[3]:


def getFilesFromFreddieMac(payload, startY, endY, file_type, target_folder):
    
    # Define the URL to be traversed to download the files.
    url='https://freddiemac.embs.com/FLoan/secure/auth.php'
    postUrl='https://freddiemac.embs.com/FLoan/Data/download.php'
        
    # Start a session and Login.
    s = requests.Session()
    preUrl = s.post(url, data=payload)
    payload2={'accept': 'Yes','acceptSubmit':'Continue','action':'acceptTandC'}
    finalUrl=s.post(postUrl,payload2)
    
    # Scrap the list of all the zip files available.
    linkhtml =finalUrl.text
    allzipfiles=BeautifulSoup(linkhtml, "html.parser")
    ziplist=allzipfiles.find_all('a')
    
    # Define list of years for which data is to be downloaded.
    year_list = []
    start_year = startY
    end_year = endY
    
    for i in range(int(start_year),int(end_year)+1):
        year_list.append(str(i))
    
    # Create final downloadable links.
    data_links=[]
    local_path=str(os.getcwd())+"/" + target_folder

    for year in year_list:
        for li in ziplist:
            if year in li.text and file_type in li.text:
                final_link ='https://freddiemac.embs.com/FLoan/Data/' + li.get('href')
                print(final_link)
                data_links.append(final_link)
    
    # Download and unzip the files.
    for lin in data_links:
        r = s.get(lin)
        z = ZipFile(BytesIO(r.content))
        z.extractall(local_path)
        print('.', end='')


# In[4]:


def main():
    print("Starting")
    
    start_year = '2018'
    end_year = '2018'
    file_type = 'sample'
    
     # Define local target location  and should be created before job execution.
    target_folder = 'SampleInputFiles'
    
    user = 'abcdefghikjl.mnopqrstuv@gmail.com'
    password = 'password'

    payload=payloadCreation(user,password)
    getFilesFromFreddieMac(payload, start_year, end_year, file_type, target_folder)


# In[5]:


if __name__ == '__main__':

    main()


# In[ ]:




