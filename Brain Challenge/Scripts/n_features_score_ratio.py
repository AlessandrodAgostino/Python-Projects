import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
import sys
import os
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
from sklearn.model_selection import KFold as KF
from sklearn.datasets import load_boston, load_iris
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump, load
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
#Loading data and defining everything necessary for loading the data from pickle
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
results_dir = '/home/alessandro/Python/Brain Challenge/Results'
data_dir='/home/alessandro/Python/Brain Challenge/Data'

data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364
x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=True)

alphas=np.arange(0.001, 10, 0.005)
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
ridge = RidgeCV(alphas=alphas, cv=5)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
regressors = [lasso, ridge, elnet]

combinations_labels = [] #titles for the subplots in the facetgrid
pipes = []
coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(pj(results_dir,filename))
    pipes.append(pipe)
    coefs.append(list(pipe.named_steps['regressor'].coef_))
    label = "{} & {}".format(sca.__class__.__name__, reg.__class__.__name__)
    combinations_labels.append(label)

#%%
#Loading from file of the results from the grid scearching
filename = "brain_gridscearch.pkl"
loaded_grid = load(pj(results_dir, filename))
result_df = pd.DataFrame(loaded_grid.cv_results_)

#Function that extract the number of coefs >= a certain treshold
def n_feat(coefs, tresh):
    return np.sum((coefs >= tresh)*1)
# which contains the number of filtered features
result_df.insert(6,"n_filtered_features", np.vectorize(n_feat)(result_df['param_Filter__coef'],
                    result_df['param_Filter__treshold']))

filters = result_df['param_Filter'].unique()

#Insert a new column with the nice name of the combination of scaler and regressor
result_df.insert(5,"nice_name", result_df['param_Filter'].map(dict(zip(filters, combinations_labels))))
#Insert a new column with the ratio between the score and the number of features
result_df.insert(7,"score_features_ratio", result_df['mean_test_score']/result_df['n_filtered_features'])
result_df.head()

#%%

fg = plt.figure()
sns.scatterplot('n_filtered_features','mean_test_score',data=result_df,s=60, alpha=0.3)
plt.title('Score n Features Ratio', size = 15)
plt.xlabel('n of Features ')
plt.ylabel('Mean test score')
fg.savefig(pj(results_dir,'Score_Features_Ratio.png'))


fg = sns.FacetGrid(data=result_df, col='nice_name', hue='nice_name', height=4, aspect=0.9)
fg.map(plt.scatter, 'n_filtered_features','mean_test_score',s=50)
fg.fig.tight_layout()
plt.show()
plt.savefig('Score_vs_#features.png')
#%%
fg = sns.FacetGrid(data=result_df, col='nice_name', hue='nice_name', height=4, aspect=0.9)
fg.map(plt.scatter, 'n_filtered_features','score_features_ratio',s=50)
fg.set(ylim=(0, 0.02))
fg.fig.tight_layout()
plt.show()
plt.savefig('score_features_ratio.png')
