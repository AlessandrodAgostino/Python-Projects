import numpy as np
import pylab as plt
import pandas as pd
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

data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results/InSmallResults'
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

scaler = MinMaxScaler()
x_train_tran = scaler.fit_transform(x_train)
x_test_tran = scaler.fit_transform(x_test)

#Simple GPR with Matern
GPR = GaussianProcessRegressor(n_restarts_optimizer=50, kernel=Matern())
b_tresh = load(pj(results_dir, "brain_best_params_in_grid0.joblib"))
b_tresh = b_tresh['Filter__treshold']
b_tresh

best_filt = CoefFilter( b_tresh, ord_coefs[0])
x_filtered = best_filt.transform(x_train_tran)
x_filtered.shape
GPR.fit(x_filtered, y_train)

xte_filtered = best_filt.transform(x_test_tran)
y_test_pred = GPR.predict(xte_filtered)
GPRy = GaussianProcessRegressor(n_restarts_optimizer=50, kernel=Matern())


GPRy.fit(y_test_pred.reshape(-1, 1), y_test.reshape(-1, 1))
y_ = np.linspace(20,78,num=200)
y_pred = GPRy.predict(y_.reshape(-1, 1))
plt.scatter(y_pred, y_)
#%%
