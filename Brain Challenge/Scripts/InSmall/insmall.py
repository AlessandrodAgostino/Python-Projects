import numpy as np
import pandas as pd
import seaborn as sns
import time
import pylab as plt
import sys
#import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os.path import join as pj
from sklearn.linear_model import RidgeCV, LassoCV
#from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
import joblib
#Sending emails
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#Caching Functions
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
sys.path.append('../')
from CachedFeaturesFilter import CachedFeaturesFilter, location, memory
from sklearn.datasets import load_boston, load_iris
import warnings

data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'

data_train = pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                            header=0, sep='\t')
    #For local use
    #X,y = load_boston(return_X_y=True)

y = data_train['age_floor'].values
X = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values

scaler = MinMaxScaler()
alphas=np.linspace(0.001, 10, num=200)
lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=1000)

GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())
transformer = CachedFeaturesFilter(lasso, 0.01,True)


pipeline = Pipeline([('Scaler', scaler),('Filter', transformer), ('GPR', GPR)])

tresholds = np.linspace(0,0.25, num=100)
parameter_grid = {'Scaler' : scalers,
                      'Filter__regressor' : regressors,
                      'Filter__regressor': regressors,
                      'Filter__treshold_mul' : tresholds,
                      'GPR__kernel' : [RBF(), DotProduct() + WhiteKernel()]}

grid = GridSearchCV(pipeline, n_jobs=16, pre_dispatch=8, param_grid=parameter_grid, cv=5)

x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=False)
