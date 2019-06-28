#token: 0616ecf58957cbf3ca90e3420ac57435ff109442b581b92d
import numpy as np
import pylab as plt
import pandas as pd
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

sys.path.append(os.path.relpath('./'))
from CoefFilter import CoefFilter

#%%
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364
x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=False)

alphas=np.arange(0.001, 10, 0.005)
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
ridge = RidgeCV(alphas=alphas, cv=5)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
regressors = [lasso, ridge, elnet]
#%%
filename = "brain_grid0.pkl"
loaded_grid = load(pj(results_dir, filename))

#%%
#Loading from joblib files pipes item already fit and their coefficients separately
pipes = []
coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(pj(results_dir,filename))
    pipes.append(pipe)
    coefs.append(list(pipe.named_steps['regressor'].coef_))


ord_coefs = np.sort(np.abs(np.array(coefs)),axis = 1)
feat_50 = ord_coefs[:,-50] #Values that correspond to the 50 feature treshold

class CoefFilter(BaseEstimator, TransformerMixin):
    history = []
    def __init__(self, treshold, coef):
        self.treshold = treshold
        self.coef = coef

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        filter = self.coef >= self.treshold
        self.history.append(filter)
        return X[:,filter]
#%%
scaler = StandardScaler()
x_train_tran = scaler.fit_transform(x_train)
x_test_tran = scaler.fit_transform(x_test)

#Simple GPR with Matern
GPR = GaussianProcessRegressor(n_restarts_optimizer=50, normalize_y=True, kernel=Matern())
b_tresh = feat_50[4]
b_tresh
best_filt = CoefFilter( b_tresh, ord_coefs[4])

x_train_tran_filt = best_filt.transform(x_train_tran)
x_test_tran_filt = best_filt.transform(x_test_tran)

GPR.fit(x_train_tran_filt, y_train)

GPR.score(x_test_tran_filt, y_test)

#%% TRYING TO DO THE PLOT ON THE Y
y_pred_on_test = GPR.predict(x_test_tran_filt)
plt.scatter(y_test,y_pred_on_test)
#%%
GPRy = GaussianProcessRegressor(n_restarts_optimizer=50, normalize_y=True,
                                kernel=Matern())
GPRy.fit(y_pred_on_test.reshape(-1, 1), y_test)

y_ = np.linspace(18,78,num=200)[:, None]
y_pred , y_std= GPRy.predict(y_, return_std=True)

plt.scatter(y_test, y_pred_on_test)
plt.plot(y_, y_pred, 'k', lw=3, zorder=9)
plt.fill_between(y_[:, 0], y_pred - y_std,
                 y_pred + y_std,
                 alpha=0.5, color='k')
