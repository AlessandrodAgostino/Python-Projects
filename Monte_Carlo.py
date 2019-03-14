#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:26:58 2019

@author: alessandro
"""
# %%
%%time
import numpy as np

data = np.random.rand(10_000, 2,100)
extraction = np.linalg.norm(data, axis=1)
n_hit = np.sum(np.where(extraction < 1, 1, 0), axis=0)
pi_ests = n_hit*4/10000

np.mean(pi_ests)
#%%
el_vec = np.array([[1,2],[3,4], [5,6]])
np.linalg.norm(el_vec, axis=1)