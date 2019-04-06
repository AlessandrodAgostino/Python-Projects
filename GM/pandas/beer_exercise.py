#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:35:45 2019

@author: alessandro
"""
import pandas as pd
import re

#Download of the raw databases

page_nobel = "https://www.britannica.com/topic/Nobel-Prize-Winners-by-Year-1856946"
page_beer = "https://en.wikipedia.org/wiki/List_of_countries_by_beer_consumption_per_capita"
page_ussr = "https://en.wikipedia.org/wiki/Post-Soviet_states"

nobel_site_tables = pd.read_html(page_nobel)
beer_site_tables = pd.read_html(page_beer)
ussr_countries_tables = pd.read_html(page_ussr)
#%%
#list of the ussr countries
countries_in_ussr = ussr_countries_tables[0].loc[:,'Country']
before_parentesis = re.compile('([\w\s]*)[\[\(]?')
countries_in_ussr = countries_in_ussr.str.extract(before_parentesis, expand=False).iloc[:-1]

##BEER##
beer_table = beer_site_tables[0]
#Select the columns I care of
beer_table = beer_table.iloc[:,0:3]
#Changing the name of the columns
beer_table.rename(index=str, columns={"Consumption per capita [1](litres)": "Consumption",
                                      "Global rank[1]": "Rank"}, inplace=True)
regex1 = re.compile('([\w\s]+)\[*.*')
beer_table['Country'] = beer_table['Country'].str.extract(regex1, expand=False)
#Merging ussr countries ins a unique entry with their mean
beer_table.loc[beer_table['Country'].isin(countries_in_ussr) , 'Country'] = 'U.S.S.R.'
beer_table = beer_table.groupby('Country', as_index=False).mean()
#Would like to modify these two lines
beer_table.loc[beer_table['Country']=='United Kingdom', 'Country']='U.K.'
beer_table.loc[beer_table['Country']=='United States', 'Country']='U.S.'

##
#find_united = re.compile('United (\w).*')
#beer_table[beer_table['Country'].str.match(find_united)]=
#'U.'+find_united.search(?????)[1]+'.'
##    

##NOBEL##
nobel_table = nobel_site_tables[0]
#Select the columns and rows I care of:
nobel_table = nobel_table.loc[nobel_table.loc[:,'category'] == 'physics',:].iloc[:,1:4]
#Changing the name of a column
nobel_table.rename(index=str, columns={'country*': 'Country'}, inplace=True)
#Ingoring division in West/East Germany
nobel_table.loc[nobel_table['Country'].str.contains('Germany'), 'Country'] = 'Germany'
#Extracting the first Country form the line where there are two divided by a /
regex2 = re.compile('([\w\.]+).*')
nobel_table['Country'] = nobel_table['Country'].str.extract(regex2, expand=False)
#Merging ussr countries
nobel_table.loc[nobel_table['Country'].isin(countries_in_ussr) , 'Country'] = 'U.S.S.R.'
#%%
#grouping by country count
nobel_table.groupby('Country').count()






























#%%
