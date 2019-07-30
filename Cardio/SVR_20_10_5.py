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

#%%
sorted_scores = pd.read_csv(pj(cardio_dir,'Sorted_scores.csv'), sep = '\t')

features_top20 = list(sorted_scores.loc[sorted_scores['top20_scores']>=0].iloc[:,0])
features_top10 = list(sorted_scores.loc[sorted_scores['top10_scores']>=5].iloc[:,0])
features_top5 = list(sorted_scores.loc[sorted_scores['top5_scores']>=7].iloc[:,0])

X_20 = data.loc[:,features_top20]
X_10 = data.loc[:,features_top10]
X_5 = data.loc[:,features_top5]

SVR1 =SVR(kernel='poly', C=3)
cv=KF(10, shuffle=True)
scaler = StandardScaler()

pipeline = Pipeline([('scaler', scaler),('SVR', SVR1)])

pargrid1 = {'SVR__kernel' : ['poly'],
            'SVR__C': [5, 7.5, 10],
            'SVR__gamma': [0.1, 1, 3]}

list_par_grid_SVR = {'SVR__kernel': ["linear", 'poly', 'rbf'],\
                     'SVR__C': [5, 7.5, 10],\
                     'SVR__degree': [1, 3, 5],\
                     'SVR__gamma': [0.1, 1, 3]}

grid_SVR = GridSearchCV(pipeline,
                         param_grid = pargrid1,
                         n_jobs=16,
                         pre_dispatch=8,
                         cv=cv)

GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())
pipe_gpr = Pipeline([('scaler', scaler),('GPR', GPR)])
grid_GPR = GridSearchCV(pipe_gpr,
                        param_grid = {'GPR__kernel': [RBF(), Matern(), RationalQuadratic()]},
                        n_jobs=16, pre_dispatch=8, )

#Working on all features
x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=False)
grid_GPR.fit(x_train, y_train)
print(grid_GPR.best_score_)
y_pred_all = grid_GPR.predict(x_test)

y_df = pd.DataFrame({'y_test':y_test,'y_pred_all':y_pred_all})
filename="grid_GPR_all.joblib"
dump(grid_GPR, pj(cardio_dir,filename))

#%%Working on 20 features
x_train,x_test,y_train,y_test=tts(X_20, y, test_size=0.1, shuffle=False)

grid_GPR.fit(x_train, y_train)
print(grid_GPR.best_score_)
y_pred_20 = grid_GPR.predict(x_test)

y_df['y_pred_20'] = y_pred_20

filename="grid_SVR_20.joblib"
dump(grid_GPR, pj(cardio_dir,filename))

#%%Working on 10 features
x_train,x_test,y_train,y_test=tts(X_10, y, test_size=0.1, shuffle=False)
grid_GPR.fit(x_train, y_train)
print(grid_GPR.best_score_)
y_pred_10 = grid_GPR.predict(x_test)

y_df['y_pred_10'] = y_pred_10

filename="grid_SVR_10.joblib"
dump(grid_GPR, pj(cardio_dir,filename))


#%%Working on 5 features
x_train,x_test,y_train,y_test=tts(X_5, y, test_size=0.1, shuffle=False)
grid_GPR.fit(x_train, y_train)
print(grid_GPR.best_score_)
y_pred_5 = grid_GPR.predict(x_test)

y_df['y_pred_5'] = y_pred_5

filename="grid_SVR_5.joblib"
dump(grid_GPR, pj(cardio_dir,filename))


y_df.to_csv(pj('/home/STUDENTI/alessandr.dagostino2/Cardio/pred_20_10_5.csv'), sep='\t')
