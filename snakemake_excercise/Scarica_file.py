#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:21:48 2019

@author: alessandro
"""



def download(filename):
    import requests

    url_base = ("https://raw.githubusercontent.com/UniboDIFABiophysics"+
                "/programmingCourseDIFA/master/snakemake_exercise/")
    response = requests.get(url_base+filename)
    
    # Throw an error for bad status codes
    response.raise_for_status()
    
    with open(filename, 'wb') as handle:
        handle.write(response.content)
        
#%%
        
for i in range(10):
    download("transazioni_0"+str(i)+".tsv")
    
for i in range(11,50):
    download("transazioni_"+str(i)+".tsv")

#%%
download("md5sums.tsv")    
    
#%%
import csv
with open("md5sums.tsv") as tsvfile:
  md5_dict = map(dict, csv.DictReader(tsvfile, fieldnames=['File', 'md5sum'], dialect ='excel-tab'))
  for row in md5_dict:
          print(row)  
#%%
import os 
os.remove("transazioni_01.tsv")        
#%%
with open("transazioni_01.tsv", 'r') as f:
    reader = csv.reader(f,delimiter='\t')
    for row in reader:
        print(row)  
          
          
          
          
          
          
          
          
          
          
          
          
          
          
