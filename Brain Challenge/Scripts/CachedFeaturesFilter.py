import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


#Theese lines are necessary to create a cache directory, and clean it before use
location = './.cachedir'
memory = Memory(location=location, verbose=1)
memory.clear(warn=False)

class CachedFeaturesFilter(BaseEstimator, TransformerMixin):
    """docstring for FilterRidgeCoefficients
    This custom transform filter those
    features with a weight greater than a certain treshold.
    It exploit the cache memory in order to not recompute those fitting that has
    already been done.
    It requires that the passed regressor has an attribute `coef_` in which the
    coefficients are stored.
    """
    def __init__(self, scaler ,regressor, treshold):
        self.scaler = scaler #The way to scale datasets
        self.regressor = regressor #The way to regress data
        self.treshold = treshold #Where to cut the coefficients
        self.regr_coef = 0

    def fit( self, X, y = None ):
        cached_pipe = Pipeline([('scaler', self.scaler),
                                ('regressor', self.regressor)])
        cached_fit = memory.cache(cached_pipe.fit)
        cached_fit(X,y)
        self.regr_coef = cached_pipe[1].coef_ #required attribute
        self.regr_coef = np.sort(np.abs(self.regr_coef))
        return self

    def transform( self, X, y = None):
        filter = self.regr_coef > self.treshold
        return X[:,filter]


#EXAMPLE OF USE EXPLOYING THE DATA IN THE REMOTE KERNEL
"""
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from os.path import join as pj

data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
data_train = pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats= data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
X=feats
y = data_train['age_floor'].values

scaler = MinMaxScaler()
alphas=np.arange(0.001, 10.001, 0.005)
lasso=LassoCV(alphas=alphas, fit_intercept=False, max_iter=100)

transformer = CachedFeaturesFilter(scaler, lasso, 1)
start = time.time()
post_filter_feats = transformer.fit_transform(feats,y)
end = time.time()
print('\nThe function took {:.2f} s to compute.'.format(end - start))
#This should take ~ 14s

start = time.time()
post_filter_feats = transformer.transform(feats,y)
end = time.time()
print('\nThe function took {:.2f} s to compute.'.format(end - start))
#This should take ~ 0.04s
"""
