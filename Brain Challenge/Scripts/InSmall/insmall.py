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
from sklearn.model_selection import KFold as KF
from joblib import dump, load
from os.path import join as pj

#Defining all the data
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364

#Defining all the scalers and regressors necessary for the coefficient scearching
alphas=np.arange(0.001, 10, 0.005)
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
ridge = RidgeCV(alphas=alphas, cv=5)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
regressors = [lasso, ridge, elnet]

#-------------------------------------------------------------------------------
#%%
#Fitting all the combinations
pipes = []
coefs = []
times = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    X = feats
    st = time.time()
    pipe.fit(X,y)
    en = time.time()
    print('\nThe fitting of {!s:.12} + {!s:.10} took {:.2f}s to compute'.format(pipe.named_steps['scaler'],pipe.named_steps['regressor'], en -st))
    pipes.append(pipe)
    coefs.append(pipe.named_steps['regressor'].coef_)
    times.append(en-st)

times
"""
The fitting of MinMaxScaler + LassoCV(al took 34.63s to compute

The fitting of MinMaxScaler + RidgeCV(al took 721.03s to compute

The fitting of MinMaxScaler + ElasticNet took 48.19s to compute

The fitting of StandardScal + LassoCV(al took 104.95s to compute

The fitting of StandardScal + RidgeCV(al took 730.67s to compute

The fitting of StandardScal + ElasticNet took 156.05s to compute

The fitting of RobustScaler + LassoCV(al took 91.00s to compute

The fitting of RobustScaler + RidgeCV(al took 739.92s to compute

The fitting of RobustScaler + ElasticNet took 133.95s to compute
"""

#-------------------------------------------------------------------------------
#%%
#Saving pipes items after fit
for pipe in pipes:
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    dump(pipe,filename)

#-------------------------------------------------------------------------------
#%%
#Loading from joblib files pipes item already fit and their coefficients separately
pipes = []
coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(filename)
    pipes.append(pipe)
    coefs.append(list(pipe.named_steps['regressor'].coef_))

#-------------------------------------------------------------------------------
#%%
#Plotting all the different coefficients
fig = plt.figure(figsize=(20, 20))
for i,coef in enumerate(coefs):#<--- limitation
    plt.subplot(3, 3, i+1)
    coef = np.sort(np.abs(coef))
    plt.plot(coef[::-1])
    plt.axvline(x=50, color='r')
    plt.axhline(y=coef[-50], color='g', label='value = {}'.format(coef[-50]))
    plt.legend()
    plt.title("Reg Coef for {!s:.12} + {!s:.10}".format(pipes[i][0],pipes[i][1]),  fontsize = 12)
    plt.tight_layout()
fig.savefig(pj(results_dir,'NineCoefPlot.png'))

#-------------------------------------------------------------------------------
#%%
ord_coefs = np.sort(np.abs(np.array(coefs))  ,axis = 1)
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

#NOW I WILL WORK ONLY WITH THE FIRST COMBINATION
filt0 = CoefFilter(feat_50[0], ord_coefs[0])
tresh0 = np.linspace(feat_50[0], ord_coefs[0,-1], num=5)
tresh0

GPR = GaussianProcessRegressor(n_restarts_optimizer=50, kernel=Matern())

filt_GPR_0 = Pipeline([('Filter', filt0), ('GPR', GPR)])
par_grid0 = {'Filter__treshold' : tresh0}

cv=KF(10, shuffle=True)
grid0 = GridSearchCV(filt_GPR_0, n_jobs=16, pre_dispatch=8,  param_grid=par_grid0, cv=cv)
st = time.time()
grid0.fit(X,y)
en = time.time()
print("\nThe fit took {:.2f}s".format(en-st))
#-------------------------------------------------------------------------------

#Theese are the lists that allow to scearch on alle the 9 different method of tresholdind
filts = [CoefFilter(feat_50[n], ord_coefs[n]) for n in range(9))]
n_tresh = 5
treshs = [np.linspace(feat_50[n], ord_coefs[n,-1], num=n_tresh) for n in range(9)]

#with different kernels
list_par_grid_multi_kernel = [{'Filter': filts[n],
                               'Filter__coef':ord_coefs[n],
                               'Filter__treshold': treshs[n],
                               'GPR__kernel': [RBF(), Matern(), RationalQuadratic()]} for n in range(9)]
#only with one kernel (Matern)
list_par_grid_same_kernel = [{'Filter': filts[n],
                               'Filter__coef':ord_coefs[n],
                               'Filter__treshold': treshs[n]} for n in range(9)]

GPR = GaussianProcessRegressor(n_restarts_optimizer=50, kernel=Matern())
cv=KF(10, shuffle=True)
pipe = Pipeline([('Filter', filts[0]), ('GPR', GPR)])
cv=KF(10, shuffle=True)

x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=False)

one_kernel_grid = GridSearchCV(pipe, param_grid = list_par_grid_same_kernel, n_jobs=16, pre_dispatch=8,  cv=cv)
st = time.time()
one_kernel_grid.fit(x_train,y_train)
en = time.time()
print("\nThe fit took {:.2f}s".format(en-st))

#Saving history of filtering
history_df = pd.DataFrame(filts[0].history)
history_df.to_csv(pj(results_dir, "brain_history_df.csv"),index = False, header = False)

#Saving best_params_ found by the gridscearch
filename = "brain_best_params_in_gridscearch.joblib"
joblib.dump(one_kernel_grid.best_params_,pj(results_dir, filename))

#Saving all the grid found by the gridscearch
filename = "brain_gridscearch.joblib"
joblib.dump(one_kernel_grid,pj(results_dir, filename))


#%%
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
