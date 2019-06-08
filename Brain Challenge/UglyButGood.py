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

#Definition of my custom Tranformer
class FilterRidgeCoefficients(BaseEstimator, TransformerMixin):
    """docstring for FilterRidgeCoefficients
        This custom transformer filter those features with a weight greater than a certain treshold."""
    def __init__(self, ridge_coefs, treshold):
        #previuous ridge regression's coefficient array
        self.ridge_coefs = ridge_coefs
        #treshold for filtering
        self.treshold = treshold

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None):
        filter = ridge_coefs > self.treshold
        return X[:,filter]

def main():
    #Remote path for Data and Results directories
    data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
    results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'

    #Loading the data as a pd.DataFrame
    data_train = pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                            header=0, sep='\t')
    #Extracting target variable as a np.array
    y = data_train['age_floor'].values
    #Extracting interesting features as a np.array
    feats= data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
    #Shape of data:
    y.shape
    feats.shape

    #Scaling of data according to the MinMaxScaler
    train_feats=MinMaxScaler().fit_transform(feats)
    #Range of values where to seek the best alpha
    alphas=np.arange(0.001, 10, 0.005)
    ridge=RidgeCV(alphas=alphas, fit_intercept=True)
    #Finding the best penalization parameter alpha
    ridge.fit(train_feats, y)
    alpha = ridge.alpha_
    alpha
    #Saving all the coefficients
    ridge_coefs = np.sort(np.abs(ridge.coef_))
    ridge_coefs.shape
    #Score of the best ridge regression:
    ridge.score(train_feats,y)

    #Defining all the tresholds where to filter coefficients
    tresholds = np.linspace(np.min(ridge_coefs), np.max(ridge_coefs)/4, num=10)

    #Definition of:
    #the filter tranformer
    filter_ridge_coef = FilterRidgeCoefficients(ridge_coefs, 0)
    #the GPR
    gaussian_process = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())
    #the pipeline that merge them
    pipe_filter_gpr = Pipeline([('filter', filter_ridge_coef), ('GPR', gaussian_process)])
    #the parameter grid with different tresholds and different kernels to try
    parameter_grid = {'filter__treshold': tresholds,
                      'GPR__kernel': [RBF(), DotProduct() + WhiteKernel()]}
    #Gridscearch of the pipeline on those parameters
    grid = GridSearchCV(pipe_filter_gpr, param_grid=parameter_grid, cv=5)

    #Splitting the data:
    x_train,x_test,y_train,y_test=tts(feats, y, test_size=0.1, shuffle=False)

    #Scearching for the best estimator:
    grid.fit(x_train,y_train)

    with open(pj(results_dir,'writing_test.txt'), 'w') as the_file:
        the_file.write('Result of the run of:\n')
        the_file.write(str(grid))
        the_file.write('\n\nBest parameters were:\n')
        the_file.write(str(grid.best_params_))
        the_file.write('\n\nBest score was\n')
        the_file.write(str(grid.best_score_))
        the_file.write('\n\nPrediction on the test set is:\n')
        the_file.write(str(grid.score(x_test,y_test)))
        the_file.write('\n')

if __name__ == '__main__':
    main()

# %%
