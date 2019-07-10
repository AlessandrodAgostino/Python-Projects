#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: carlomengucci

# %% ### Prostate cancer model selection GridSearch ###

import numpy as np

import scipy.stats as st
import pandas as pd
import os
from os.path import join as pj

import seaborn as sns
import pylab as plt

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import LeaveOneOut

#from sklearn.ensemble import enable_hist_gradient_boosting
# from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split as tts
# token=696cdee56b1bfdbd084b53c061a678bf8dc3b31241e14490

## Remote path
# data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
# results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'

## Local path
data_dir='/home/alessandro/Dropbox/UniBo/Brain Challenge/Data'
results_dir='/home/alessandro/Dropbox/UniBo/Brain Challenge/Results'

# %% ## Train DataFrame Loading ##
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')


data_train.head()
