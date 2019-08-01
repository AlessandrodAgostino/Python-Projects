import numpy as np
import pylab as plt
import pandas as pd
from itertools import product
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from os.path import join as pj
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold as KF
from joblib import dump, load
from os.path import join as pj


#%%
cardio_dir='/home/STUDENTI/alessandr.dagostino2/Cardio'
data = load(pj(cardio_dir,"dati.pickle"))

not_num_features = ["age","AA", "RR", "Rpeakvalues",
                    "rhythm", "sex","signal","time",
                    "mean_time","mean_signal","var_mean_signal"]

features = data.loc[:,[i for i in data.columns if i not in not_num_features]]
features.apply(lambda col:pd.to_numeric(col, errors='coerce'))
X = features.values
y = data["age"].values

y_df.to_csv(pj('/home/STUDENTI/alessandr.dagostino2/Cardio/pred_20_10_5.csv'), sep='\t')
