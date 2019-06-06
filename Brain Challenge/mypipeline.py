#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from os.path import join as pj

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import GridSearchCV
import pylab as plt
from sklearn.model_selection import train_test_split as tts


# Kernel: fba6b7952e314279cfb048a5132470bc1f167f76bd6a7904


## Remote path
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
## Local path
#data_dir='/home/alessandro/Dropbox/UniBo/Brain Challenge/Data'
#results_dir='/home/alessandro/Dropbox/UniBo/Brain Challenge/Results'

data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
#data_train.head()
feats= data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y=data_train['age_floor'].values

#Scaled feat
train_feats=MinMaxScaler().fit_transform(feats)
alphas=np.arange(0.001, 10, 0.005)
ridge=RidgeCV(alphas=alphas, fit_intercept=False, store_cv_values=True)
ridge.fit(train_feats,y)
ridge.alpha_
ridge.score(train_feats, y)

#Why are they only 954 if there are 2364 features?
len(ridge.coef_)
len(train_feats)

plt.plot(np.sort(np.abs(ridge.coef_)))
