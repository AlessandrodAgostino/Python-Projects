import numpy as np
import pandas as pd
import seaborn as sns
import time
import pylab as plt
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
from CachedFeaturesFilter import CachedFeaturesFilter, location, memory
from Pipeline_GridscearchCV import data_dir, results_dir

from sklearn.datasets import load_boston, load_iris
import warnings


#best_params = joblib.load(pj(results_dir,'AstroTOP100_best_est_gs.pkl'))
best_params = joblib.load("/home/alessandro/Python/Brain Challenge/Results/best_params_in_gridscearch.pkl")
dumped_grid = joblib.load("/home/alessandro/Python/Brain Challenge/Results/best_params_in_gridscearch.pkl")

scaler = MinMaxScaler()
scalers  = [MinMaxScaler(), StandardScaler()]
alphas=np.linspace(0.001, 10, num=200)
lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=1000)
ridge=RidgeCV(alphas=alphas, fit_intercept=True)
regressors = [lasso, ridge]
GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=3, kernel=RBF())
transformer = CachedFeaturesFilter(lasso, 0.01,True)
pipeline = Pipeline([('Scaler', scaler),('Filter', transformer), ('GPR', GPR)])
tresholds = np.linspace(0,0.25, num=100)
parameter_grid = {'Scaler' : scalers,
                      'Filter__regressor' : regressors,
                      'Filter__treshold_mul' : tresholds,
                      'GPR__kernel' : [RBF(), DotProduct() + WhiteKernel()]}
grid = GridSearchCV(pipeline, param_grid=parameter_grid, cv=2)

pipeline.set_params(**best_params)
#To control if it really it's so easy!!
grid = dumped_grid

X,y = load_boston(return_X_y=True)

pipeline.predict(X)
