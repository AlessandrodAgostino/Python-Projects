import numpy as np
import pylab as plt
import pandas as pd
from itertools import product
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from os.path import join as pj
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from joblib import load
#%%
#------------------------------------------------------------------------------
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')

#defining everything needed for loading from pickles
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
#------------------------------------------------------------------------------
#%%
#Computing the scores for each features
coefs = np.array(coefs)
ord_coef = np.sort(np.abs(coefs), axis=1)

top50_scores = np.zeros(*coefs[0].shape)
top25_scores = np.zeros(*coefs[0].shape)
top10_scores = np.zeros(*coefs[0].shape)

for i in range(9):
    top50_scores += (np.abs(coefs[i]) >= ord_coef[i,-50])*1
    top25_scores += (np.abs(coefs[i]) >= ord_coef[i,-25])*1
    top10_scores += (np.abs(coefs[i]) >= ord_coef[i,-10])*1
#%%
#Writing the dataframe with the scores of different top  ranks
scores_df = pd.DataFrame([top50_scores,top25_scores,top10_scores],
                         columns = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].columns)

scores_df.set_index(pd.Index(["top50_scores","top25_scores","top10_scores"]))
scores_df = scores_df.T
scores_df.columns = ['top50_scores', 'top25_scores', 'top10_scores']
sort_score_df = scores_df.sort_values(by=['top10_scores'], ascending= False)
sort_score_df = sort_score_df.query("top50_scores > 0")
sort_score_df.to_csv(pj(results_dir,'Sorted_scores.csv'))
