#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:36:01 2019

@author: alessandro
"""

import pandas as pd
import numpy as np

page = 'https://en.wikipedia.org/wiki/List_of_highest-grossing_films'
wikitables = pd.read_html(page)
#%%
wikitables = pd.read_html(page)
dataframe = wikitables[0].copy()


#%%
dataframe.columns