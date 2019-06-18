from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

location = './cachedir'
memory = Memory(location=location, verbose=1)
memory.clear(warn=False) #Command used to clear the memory

class CachedFeaturesFilter(BaseEstimator, TransformerMixin):
    def __init__(self, regressor, treshold_mul , selection_history = False, history = None):
        self.regressor = regressor #The way to regress data
        self.treshold_mul = treshold_mul #Where to cut the coefficients
        self.filter = 0
        self.selection_history = selection_history
        self.history = history

    def fit( self, X, y = None ):
        #Here I cache the fit method of the local pipeline, that will be called below
        cached_fit = memory.cache(self.regressor.fit)
        cached_fit(X,y)
        if hasattr(self.regressor, 'coef_'):
            regr_coef = self.regressor.coef_
        else:
            regr_coef = np.zeros(len(X[0,:]))

            #Here I save the regression coefficients in absolut value
            regr_coef = np.abs(regr_coef)
            #Here I seek the Min and the Max of the coefficients and set the treshold
            min_coef = np.min(regr_coef)
            delta = np.max(regr_coef) - min_coef
            self.filter = regr_coef >= min_coef + delta*self.treshold_mul
        return self

    def transform(self, X, y = None):
        #Saving the history of filtering
        if self.selection_history: self.history.append(filter*1)
        return X[:,self.filter]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X,y).transform(X)
