import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from os.path import join as pj
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

#
# GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())
# GPR.fit(X,y)
# GPR.score(X,y)
#
# param_grid = { 'kernel' : [RBF(), DotProduct() + WhiteKernel()]}
# grid = GridSearchCV(GPR, param_grid = param_grid)
# grid.fit(X,y)
# grid.best_estimator_
# grid.cv_results_
# grid.score(X,y)

class UselessTransform(BaseEstimator, TransformerMixin):
    def __init__(self,foo):
        self.foo = foo
    #Return self nothing else to do here
    def fit( self, X, y = None ):
        return self

    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X

class FilterRidgeCoefficients(BaseEstimator, TransformerMixin):
    def __init__(self, ridge_coefs, treshold):
        self.ridge_coefs = ridge_coefs
        self.treshold = treshold

    def fit( self, X, y = None ):
        return self

    def transform( self, X, tresh=0.0, y = None):
        filter = ridge_coefs > tresh
        return X[:,filter]

# useless = UselessTransform(4)
pipe = Pipeline([('nothing', useless),
                ('GPR', GPR)])
# pipe.get_params().keys()
# pipe.fit(X,y)
# pipe.score(X,y)

param_grid2 = { 'nothing__foo': [1,2],
                'GPR__kernel': [RBF(), DotProduct() + WhiteKernel()]}
#param_grid2 = {'GPR__kernel': [RBF()]}

grid2 = GridSearchCV(pipe, param_grid= param_grid2)
grid2.fit(X,y)
grid2.best_params_
grid2.score(X,y)
grid2.best_score_
print(grid2)

with open('/home/alessandro/Python/Brain Challenge/writing_test.txt', 'w') as the_file:
    the_file.write('Result of the run of:\n')
    the_file.write(str(grid2))
    the_file.write('\n\nBest parameters were:\n')
    the_file.write(str(grid2.best_params_))
    the_file.write('\n\nBest score was\n')
    the_file.write(str(grid2.best_score_))
    the_file.write('\n\nPrediction on the test set is:\n')
    the_file.write(str(grid2.score(X,y)))
    the_file.write('\n')
