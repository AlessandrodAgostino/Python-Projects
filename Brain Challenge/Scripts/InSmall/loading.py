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
from sklearn.datasets import load_boston, load_iris
from sklearn.base import BaseEstimator, TransformerMixin

from joblib import dump, load

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
    pipe = load(filename)
    pipes.append(pipe)
    coefs.append(pipe.named_steps['regressor'].coef_)

#%%
#Plotting all the different coefficients
fig = plt.figure(figsize=(20, 20))
for i,coef in enumerate(coefs):
    plt.subplot(3, 3, i+1)
    plt.plot(np.sort(np.abs(coef))[::-1])
    plt.title("Reg Coef for {!s:.12} + {!s:.10}".format(pipes[i][0],pipes[i][1]),  fontsize = 12)
    plt.tight_layout()
fig.savefig('./NineCoefPlot.png')
