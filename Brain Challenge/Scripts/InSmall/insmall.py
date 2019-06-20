import numpy as np
import pylab as plt
import pandas as pd
from itertools import product

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from os.path import join as pj
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston, load_iris

data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats= data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values

y=data_train['age_floor'].values
X = feats
#n_features = 954
#n_samples = 2364
alphas=np.arange(0.001, 10, 0.005)
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
lasso = LassoCV(alphas=alphas, max_iter=100000)
ridge = RidgeCV(alphas=alphas)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000)
regressors = [lasso, ridge, elnet]

coefs = []
for sca, reg in product(scalers, regressors):
    X = feats
    X = sca.fit_transform(X)
    reg.fit(X,y)
    coefs.append(reg.coef_)

# #%%
#
# scaler = MinMaxScaler()
#
# alphas=np.arange(0.001, 10, 0.005)
# lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=100000)
#
# pipe = Pipeline([('scaler', scaler), ('lasso', lasso)])
#
# x_train,x_test,y_train,y_test=tts(X, y, test_size=0.4, shuffle=False)
#
# pipe.fit(x_train, y_train)
# y_pr = pipe.predict(x_test)
#
# GPRy=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=Matern())
# GPRy.fit(y_pr[:,None], y_test[:,None])
# y_ = np.linspace(5,50,num=100)
# y_print, y_std = GPRy.predict(y_[:,None], return_std=True)
# plt.plot(y_print,y_)
s
