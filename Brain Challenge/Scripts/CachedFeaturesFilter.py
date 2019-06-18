from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

#To import or to recreate in the destination file
# location = './cachedir'
# memory = Memory(location=location, verbose=1)
# memory.clear(warn=False) #Command used to clear the memory



class CachedFeaturesFilter(BaseEstimator, TransformerMixin):
    """
    The method exploit the cache memory using `joblib` in order to not recompute
    those `fit` that have already been done. In order to work wel it's required
    to create a cachememory directory as follows:

        location = './cachedir'
        memory = Memory(location=location, verbose=1)
        memory.clear(warn=False) #Command used to clear the memory

    This custom transformer takes as parameters a datasets `X `and the targets
    `y` and returns only those features that have a coefficient greater than a
    certain treshold.

    The treshold is intended as a multiplier of the gap between the MIN and the
    MAX of the coefficients array:

        effective_tresh = (MAX_COEF - MIN_COEF)*tresh_multiplier + MIN_COEF

    The history of the different filtering that have been done can be stored in
    a list `history` that's an attribute of the whole class. In order to do that
    the attribute `selection_history` should be set to `True`.

    #EXAMPLE OF USE:
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import time
from sklearn.pipeline import Pipeline
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    location = './cachedir'
    memory = Memory(location=location, verbose=1)
    memory.clear(warn=False) #Command used to clear the memory

    #Load data and target as np.array
    X, y = load_boston(return_X_y=True)

    scaler = MinMaxScaler()
    alphas = np.linspace(1e-4, 1e2, num = 200)
    lasso = LassoCV(alphas=alphas, fit_intercept=False, max_iter=1000)
    transformer = CachedFeaturesFilter(lasso, 0.01, selection_history = True)

    #Let's see the use of the cache:
    start = time.time()
    transformer.fit_transform(X,y)
    end = time.time()
    print('\nThe function took {:.2f} s to compute.'.format(end - start))

    start = time.time()
    transformer.fit_transform(X,y)
    end = time.time()
    print('\nThe function took {:.2f} s to compute.'.format(end - start))

    start = time.time()
    transformer.fit_transform(X,y)
    end = time.time()
    print('\nThe function took {:.2f} s to compute.'.format(end - start))


    #In this tiny df the difference is negligible but, this first run took ~0.15s
    #From the second run on the time spent is ~0.02s

    #It works well also inside a pipeline
    GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())
    pipeline = Pipeline([('scaler', scaler), ('filter', transformer),('GPR',GPR)])
    pipeline.fit(X,y)
    print("The history of the filtering is:",transformer.history)
    """

    history = []
    def __init__(self, regressor, treshold_mul , selection_history = False):
        self.regressor = regressor #The way to regress data
        self.treshold_mul = treshold_mul #Where to cut the coefficients
        self.selection_history = selection_history

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
        self.filter = regr_coef >= (min_coef + delta*self.treshold_mul)
        return self

    def transform(self, X, y = None):
        #Saving the history of filtering
        if self.selection_history:
            self.history.append(self.filter*1)
        return X[:,self.filter]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X,y).transform(X)
