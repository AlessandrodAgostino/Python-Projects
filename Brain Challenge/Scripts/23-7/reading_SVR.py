#b54a71e1422cbed7397b4d94d63fb01edc038a35e6f1974f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pj
from joblib import dump, load
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin



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
#Loading data

data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364

y_df=pd.read_csv(pj(results_dir,'y_df'), sep='\t')

filename="grid_SVR.joblib"
grid_SVR = load(pj(results_dir,'joblib',filename))

grid_SVR.fit(X,y)
#%%

grid_SVR.best_score_
y_df.head()

y_test = y_df['y_test'].values
y_pred = y_df['y_pred'].values
#%%
g = sns.lmplot(x = 'y_test', y ='y_pred', data =  y_df, robust = True, height = 10)
#%%
g = sns.lmplot(x = 'y_test', y ='y_pred', data =  y_df, logx = True, height = 10)
#%%
slope, intercept, r_value, p_value, std_err = stats.linregress(y_df['y_test'], y_df['y_pred'])
r_value**2
g = sns.lmplot(x = 'y_test', y ='y_pred', data =  y_df, height = 9)
g.fig.suptitle('Linear regression $R^2$={:.2f}'.format(r_value**2))
#%%
fig = plt.figure()
plt.scatter(y_test,y_pred)
plt.gca().set_title("Predictor's score: {:.3f}".format(grid_SVR.best_score_, size = 17))
plt.gca().set_ylabel("Predicted age", size=15)
plt.gca().set_xlabel("Age", size=15)
fig.savefig(pj(results_dir, '24-7_SVR.png'), bbox_inches='tight')
