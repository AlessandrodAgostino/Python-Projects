#token: 409812b6032c99a44403e4db39834192ce7dd38ff17a907c
import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
import sys
import os
from itertools import product
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from os.path import join as pj
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold as KF
from sklearn.datasets import load_boston, load_iris
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin

class CoefFilter(BaseEstimator, TransformerMixin):
    history = []
    def __init__(self, treshold, coef):
        self.treshold = treshold
        self.coef = coef

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        filter = self.coef >= self.treshold
        self.history.append(list(filter))
        return X[:,filter]

#%%
#Loading data and defining everything necessary for loading the data from pickle
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364

alphas=np.arange(0.001, 10, 0.005)
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
ridge = RidgeCV(alphas=alphas, cv=5)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
regressors = [lasso, ridge, elnet]

coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(pj(results_dir,filename))
    coefs.append(list(pipe.named_steps['regressor'].coef_))
#%%
#Loading from file of the results from the grid scearching
filename = "brain_gridscearch.pkl"
loaded_grid = load(pj(results_dir, filename))

#%%
best_pipe = loaded_grid.best_estimator_


x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=True)
best_pipe.fit(x_train,y_train)
best_pipe.score(x_test,y_test)

#%%
fig = plt.figure()
y_pred_on_test = best_pipe.predict(x_test)
plt.scatter(y_test,y_pred_on_test)
plt.gca().set_ylabel("y predicted on x_test", size=15)
plt.gca().set_xlabel("y_test", size=15)
fig.savefig(pj(results_dir, 'After_train_without_fit.png'), bbox_inches='tight')

#%% TRYING TO DO THE PLOT ON THE Y

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha=0.0)
gp.fit(y_pred_on_test.reshape(-1, 1), y_test)

GPRy = GaussianProcessRegressor(n_restarts_optimizer=50, normalize_y=True, kernel=Matern())
GPRy.fit(y_pred_on_test.reshape(-1, 1), y_test)

y_ = np.linspace(18,78,num=200)[:, None]
y_pred , y_std= gp.predict(y_, return_std=True)

fig = plt.figure()
y_pred_on_test = best_pipe.predict(x_test)
plt.scatter(y_test,y_pred_on_test)
plt.gca().set_ylabel("y predicted on x_test", size=15)
plt.gca().set_xlabel("y_test", size=15)
plt.gca().plot(y_, y_pred, 'k', lw=3, zorder=9)
plt.gca().fill_between(y_[:, 0], y_pred - y_std,
                 y_pred + y_std,
                 alpha=0.5, color='k')
fig.savefig(pj(results_dir, 'After_train_with_fit1.png'), bbox_inches='tight')
