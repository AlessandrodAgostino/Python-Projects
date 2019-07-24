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
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold as KF
from joblib import dump, load
from os.path import join as pj


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
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
scripts_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Scripts'

data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364

#%%
sorted_scores = pd.read_csv(pj(scripts_dir, '24-7','Sorted_scores.csv'))
features_top50 = list(sorted_scores.loc[sorted_scores['top50_scores']>=0].iloc[:,0])
features_top25 = list(sorted_scores.loc[sorted_scores['top50_scores']>=5].iloc[:,0])
features_top10 = list(sorted_scores.loc[sorted_scores['top50_scores']>=7].iloc[:,0])

X_50 = data_train.loc[:,features_top50]
X_25 = data_train.loc[:,features_top25]
X_10 = data_train.loc[:,features_top10]

SVR1 =SVR(kernel='linear', C=3)
cv=KF(10, shuffle=True)
scaler = StandardScaler()

pipeline = Pipeline([('scaler', scaler),('SVR', SVR1)])
list_par_grid_SVR = {'SVR__kernel': ["linear", 'poly', 'rbf', 'sigmoid'],\
                     'SVR__C': [5, 6.25, 7.5, 8.75, 10],\
                     'SVR__degree': [1, 2, 3, 4, 5, 6],\
                     'SVR__gamma': [0.1, 0.5,1,2.5,3]}

grid_SVR = GridSearchCV(pipeline,
                         param_grid = list_par_grid_SVR,
                         n_jobs=16,
                         pre_dispatch=8,
                         cv=cv)

#%%Working on 50 features
x_train,x_test,y_train,y_test=tts(X_50, y, test_size=0.1, shuffle=False)

grid_SVR.fit(x_train, y_train)
y_pred_50 = grid_SVR.predict(x_test)

y_df = pd.DataFrame({'y_test_50':y_test,'y_pred_50':y_pred_50})
filename="grid_SVR_50.joblib"
dump(grid_SVR, pj(scripts_dir,'24-7',filename))

#%%Working on 25 features
x_train,x_test,y_train,y_test=tts(X_25, y, test_size=0.1, shuffle=False)

grid_SVR.fit(x_train, y_train)
y_pred_25 = grid_SVR.predict(x_test)

y_df['y_test_25'] = y_test
y_df['y_pred_25'] = y_pred_25

filename="grid_SVR_25.joblib"
dump(grid_SVR, pj(scripts_dir,'24-7',filename))

#%%Working on 10 features
x_train,x_test,y_train,y_test=tts(X_10, y, test_size=0.1, shuffle=False)

grid_SVR.fit(x_train, y_train)
y_pred_10 = grid_SVR.predict(x_test)

y_df['y_test_10'] = y_test
y_df['y_pred_10'] = y_pred_10

filename="grid_SVR_10.joblib"
dump(grid_SVR, pj(scripts_dir,'24-7',filename))
