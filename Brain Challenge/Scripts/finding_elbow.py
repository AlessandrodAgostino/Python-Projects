import numpy as np
import pylab as plt
import pandas as pd
from itertools import product
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from os.path import join as pj
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold as KF
from joblib import dump, load
from os.path import join as pj
from sklearn.pipeline import Pipeline

#%%
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
scripts_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Scripts'

data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364

scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
alphas=np.arange(0.001, 10, 0.005)
lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
ridge = RidgeCV(alphas=alphas, cv=5)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
regressors = [lasso, ridge, elnet]


#%% Loading
pipes = []
coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(pj(results_dir,filename))
    pipes.append(pipe)
    coefs.append(list(pipe.named_steps['regressor'].coef_))

#%%
#Computing the scores for each features
coefs = np.array(coefs)
ord_coef = np.sort(np.abs(coefs), axis=1)
