#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: carlomengucci

# %% ### Prostate cancer model selection GridSearch ###

import numpy as np

import scipy.stats as st
import pandas as pd
import os
from os.path import join as pj

import seaborn as sns
import pylab as plt

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import LeaveOneOut

#from sklearn.ensemble import enable_hist_gradient_boosting
# from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split as tts
# token=696cdee56b1bfdbd084b53c061a678bf8dc3b31241e14490

## Remote path
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'

## Local path
#data_dir='/home/alessandro/Dropbox/UniBo/Brain Challenge/Data'
#results_dir='/home/alessandro/Dropbox/UniBo/Brain Challenge/Results'

# %% ## Train DataFrame Loading ##
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
data_train.head()

feats= data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values

y=data_train['age_floor'].values



# %% ###### GRIDSEARCH #######

#### Scalers
scalers=[RobustScaler(), StandardScaler(), MinMaxScaler()]

### Regressions
#cvR=SKF(10).split(feats, data_train['gender'])

#a=list(cvR)

alphas=np.arange(0.001, 10, 0.005)

lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=100000)
ridge=RidgeCV(alphas=alphas, fit_intercept=True)
pls=PLSRegression(n_components=10, scale=False, max_iter=1000)
gbr=GradientBoostingRegressor(loss='lad', alpha=0.7)

SVR=SVR(kernel='linear', C=3)
GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())

regressors=[lasso,ridge]

# %% #### Pipeline ####

#cv=SKF(10).split(feats, data_train['site'])
cv=KF(10, shuffle=True)

pipe = Pipeline([('scale', StandardScaler()),
                ('regress', lasso)])

param_grid = [{ 'scale': scalers,
                'regress': regressors}]

grid = GridSearchCV(pipe, n_jobs=16, pre_dispatch=8,  param_grid=param_grid, cv=cv)
grid.fit(feats, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores

best_est=grid.cv_results_['params'][grid.best_index_]

best_est

grid.cv_results_

### N.B score is R^2
grid.best_score_


# %% ### Prediction ###
test_data=pd.read_csv(pj(data_dir,
'Data-eTIV/Train_Test_NOremove14_NOremoveOutliers_YESregressBYeTIVifCorr_20190528/Test_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')

test_data.head()

test_feats= test_data.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values



train_feats=MinMaxScaler().fit_transform(feats)
test_feats=MinMaxScaler().fit_transform(test_feats)

alphas=np.arange(0.001, 10, 0.005)

ridge=RidgeCV(alphas=alphas, fit_intercept=True)



# %% ### Penalized Model ####
for i in np.arange(0,10, 0.5):
    x_train,x_test,y_train,y_test=tts(train_feats, y, test_size=0.1, shuffle=True)

    ridge.fit(x_train,y_train)
    coefs=ridge.coef_

    coef_f=np.abs(coefs) > i

    red_feats=x_train[:,coef_f]

    ### Reduced Feats model prediction

    ridge.fit(red_feats, y_train)

    print(ridge.score(x_test[:,coef_f], y_test))

# %%
### Train model ##
ridge.fit(train_feats,y)

## Predict Test set ##
age_predicted=np.floor(ridge.predict(test_feats))

age_predicted


# %% ### Train Model Plotting ###
ridge.fit(train_feats,y)

train_prediction=np.floor(ridge.predict(train_feats))
train_score=ridge.score(train_feats,y)

train_score

data_train['predicted_age'] = train_prediction

f=sns.lmplot('age_floor', 'predicted_age',data=data_train, robust=True,
            scatter_kws={'alpha':0.2}, hue='gender', height=8, ci=90)
plt.gca().set_title(r'Final Model Full Train Result, $R^2=$%.2f'%train_score, size=15)
plt.gca().set_ylabel('Predicted Age', size=15)
plt.gca().set_xlabel('Age', size=15)
f.savefig(pj(results_dir, 'Full_Train_final_Result_gender.png'), bbox_inches='tight')

# %% ### Datframe Creation ##
age_sub=dict(zip(test_data['subject_ID'], age_predicted))

age_sub

age_df=pd.DataFrame.from_dict(data=age_sub, orient='index',columns=['predicted_age'])

age_df.to_csv(pj(results_dir, 'Test_Prediction.csv'), header=True, sep=',')
