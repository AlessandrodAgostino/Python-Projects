import numpy as np
import pylab as plt
import pandas as pd
from itertools import product
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from os.path import join as pj
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold as KF
from joblib import dump, load
from os.path import join as pj
from sklearn.pipeline import Pipeline

#%%
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
scripts_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Scripts'

data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364

scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
alphas=np.arange(0.001, 10, 0.005)
lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
ridge = RidgeCV(alphas=alphas, cv=5)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
regressors = [lasso, ridge, elnet]

#%% Loading
pipes = []
coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(pj(results_dir,'joblib',filename))
    pipes.append(pipe)
    coefs.append(list(pipe.named_steps['regressor'].coef_))

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
fig = plt.figure(figsize=(20, 20))
for i,coef in enumerate(coefs):
    plt.subplot(3, 3, i+1)
    coef = np.sort(np.abs(coef))
    plt.plot(coef[::-1], )
    plt.axvline(x=50, color='r')
    plt.axhline(y=coef[-50], color='g', label='50 feat threshold = {:.3}'.format(coef[-50]))
    plt.legend(fontsize=12)
    plt.title("Reg Coef for {!s:.12} + {!s:.10}".format(str(pipes[i][0]).split("(")[0],str(pipes[i][1]).split("CV")[0]),  fontsize = 15)
    plt.tight_layout()
plt.savefig(pj(scripts_dir,'24-7','NineCoefPlot.png'), bbox_inches='tight')

#%%
#Writing the dataframe with the scores of different top  ranks
scores_df = pd.DataFrame([top50_scores,top25_scores,top10_scores],
                         columns = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].columns)
scores_df = scores_df.T
scores_df.columns = ['top50_scores', 'top25_scores', 'top10_scores']
sort_score_df = scores_df.sort_values(by=['top10_scores'], ascending= False)
sort_score_df = sort_score_df.query("top50_scores > 0")
sort_score_df.to_csv(pj('/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Scripts/24-7/Sorted_scores.csv'))

sort_score_df = scores_df.sort_values(by=['top10_scores'], ascending= False)
sort_score_df
