import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from os.path import join as pj
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import KFold as KF
from joblib import dump, load

#%%
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
scripts_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Scripts'

data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats
#n_features = 954 n_samples = 2364

#%% Loading previously computed scores
sorted_scores = pd.read_csv(pj(scripts_dir,  'Sorted_scores.csv'))
features_top50 = list(sorted_scores.loc[sorted_scores['top50_scores']>0].iloc[:,0])
features_top25 = list(sorted_scores.loc[sorted_scores['top25_scores']>0].iloc[:,0])
features_top10 = list(sorted_scores.loc[sorted_scores['top10_scores']>0].iloc[:,0])

X_50 = data_train.loc[:,features_top50]
X_25 = data_train.loc[:,features_top25]
X_10 = data_train.loc[:,features_top10]

#%% Defining regression tools
SVR1 =SVR(kernel='poly', C=3)
cv=KF(10, shuffle=True)
scaler = StandardScaler()

pipeline = Pipeline([('scaler', scaler),('SVR', SVR1)])

"""
Alternative Parameter Grid:
pargrid1 = {'SVR__kernel' : ['poly'],
            'SVR__C': [5, 7.5, 10],\
            'SVR__degree': [1, 3, 5],\
            'SVR__gamma': [0.1, 1, 3]}
"""

list_par_grid_SVR = {'SVR__kernel': ['linear','rbf'],
                     'SVR__C': [5, 7.5, 10],
                     'SVR__gamma': [0.1, 1, 3]}

grid_SVR = GridSearchCV(pipeline,
                         param_grid = list_par_grid_SVR,
                         n_jobs=16,
                         pre_dispatch=8,
                         cv=cv)

#Working on the different cuts:
#%%Working on 50 features
#Diving the dataset in training and testing set
x_train,x_test,y_train,y_test=tts(X_50, y, test_size=0.1, shuffle=False)
#Fitting the training dataset
grid_SVR.fit(x_train, y_train)
#Predicting on test dataset
y_pred_50 = grid_SVR.predict(x_test)
#Saving the predictions
y_df = pd.DataFrame({'y_test':y_test,'y_pred_50':y_pred_50})
#Saving the results of fitting for future use
filename="grid_SVR_50.joblib"
dump(grid_SVR, pj(scripts_dir,filename))

#%%Working on 25 features
x_train,x_test,y_train,y_test=tts(X_25, y, test_size=0.1, shuffle=False)

grid_SVR.fit(x_train, y_train)
y_pred_25 = grid_SVR.predict(x_test)

y_df['y_pred_25'] = y_pred_25

filename="grid_SVR_25.joblib"
dump(grid_SVR, pj(scripts_dir,filename))

#%%Working on 10 features
x_train,x_test,y_train,y_test=tts(X_10, y, test_size=0.1, shuffle=False)

grid_SVR.fit(x_train, y_train)
y_pred_10 = grid_SVR.predict(x_test)

y_df['y_pred_10'] = y_pred_10

filename="grid_SVR_10.joblib"
dump(grid_SVR, pj(scripts_dir, filename))

#%% Exporting the prediction in .csv
y_df.to_csv(pj(scripts_dir,  'y_df_50_25_10.csv'), sep='\t', index = "False")
