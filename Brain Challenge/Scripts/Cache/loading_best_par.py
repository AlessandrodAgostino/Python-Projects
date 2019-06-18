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
from sklearn.datasets import load_boston
from CachedFeaturesFilter import CachedFeaturesFilter

scaler = MinMaxScaler()
alphas=np.arange(0.001, 10, 0.005)
lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=100)
transformer = CachedFeaturesFilter(scaler, lasso, 0.5, selection_history = False)
GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())
pipeline = Pipeline([('Filter', transformer),
                     ('GPR', GPR)])

filename = pj('toy_iris_best.pkl')
best_est = joblib.load(filename)
pipeline.get_params()
