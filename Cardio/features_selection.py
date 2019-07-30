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
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
alphas=np.arange(0.001, 10, 0.005)
lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
ridge = RidgeCV(alphas=alphas, cv=5)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
regressors = [lasso, ridge, elnet]

#Fitting all the combinations
pipes = []
coefs = []

for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    pipe.fit(X,y)
    pipes.append(pipe)
    coefs.append(pipe.named_steps['regressor'].coef_)

#Saving pipes items after fit
for pipe in pipes:
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    dump(pipe,pj(cardio_dir,filename))

#%% Loading
pipes = []
coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(pj(cardio_dir,filename))
    pipes.append(pipe)
    coefs.append(list(pipe.named_steps['regressor'].coef_))

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
plt.savefig(pj(cardio_dir,'NineCoefPlot.png'), bbox_inches='tight')

#-------------------------------------------------------------------------------
#%%
#Computing the scores for each features
coefs = np.array(coefs)
ord_coef = np.sort(np.abs(coefs), axis=1)

top20_scores = np.zeros(*coefs[0].shape)
top10_scores = np.zeros(*coefs[0].shape)
top5_scores = np.zeros(*coefs[0].shape)

for i in range(9):
    top20_scores += (np.abs(coefs[i]) >= ord_coef[i,-20])*1
    top10_scores += (np.abs(coefs[i]) >= ord_coef[i,-10])*1
    top5_scores += (np.abs(coefs[i]) >= ord_coef[i,-5])*1

#%%
#Writing the dataframe with the scores of different top  ranks
scores_df = pd.DataFrame([top20_scores,top10_scores,top5_scores],
                         columns = features.columns)
scores_df = scores_df.T
scores_df.columns = ['top20_scores', 'top10_scores', 'top5_scores']

sort_score_df = scores_df.sort_values(by=['top5_scores'], ascending= False)
#sort_score_df = sort_score_df.query("top20_scores > 0")

sort_score_df

sort_score_df.to_csv(pj('/home/STUDENTI/alessandr.dagostino2/Cardio/Sorted_scores.csv'), sep='\t')
