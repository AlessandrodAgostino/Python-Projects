import numpy as np
import pandas as pd
import seaborn as sns
import time
import pylab as plt
#import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from os.path import join as pj
from sklearn.linear_model import RidgeCV, LassoCV
#from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
#Sending emails
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#Caching Functions
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


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

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None):
        cached_pipe = Pipeline([('scaler', self.scaler),
                                ('regressor', self.regressor)])
        cached_fit = memory.cache(cached_pipe.fit)
        cached_fit(X,y)
        regr_coef = cached_pipe[1].coef_ #required attribute
        regr_coef = np.sort(np.abs(regr_coef))
        filter = regr_coef > self.treshold
        return X[:,filter]


def main():
    #Definition of my custom Tranformer
    location = './.cachedir'
    memory = Memory(location=location, verbose=1)
    memory.clear(warn=False)

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

    scaler = MinMaxScaler()
    alphas=np.arange(0.001, 10, 0.005)
    lasso=LassoCV(alphas=alphas, fit_intercept=False, max_iter=100)

    transformer = CachedFeaturesFilter(scaler, lasso, 1)
    GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())

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

 ############################################################

    start = time.time()
    post_filter_feats = transformer.transform(feats,y)
    end = time.time()
    print('\nThe function took {:.2f} s to compute.'.format(end - start))
    #This should take ~ 14s

    #Scaling of data according to the MinMaxScaler
    train_feats=MinMaxScaler().fit_transform(feats)
    #Range of values where to seek the best alpha
    alphas=np.arange(0.001, 10, 0.005)
    ridge=RidgeCV(alphas=alphas, fit_intercept=True)
    #Finding the best penalization parameter alpha
    ridge.fit(train_feats, y)
    #alpha = ridge.alpha_
    #alpha
    #Saving all the coefficients
    ridge_coefs = np.sort(np.abs(ridge.coef_))
    #ridge_coefs.shape
    #Score of the best ridge regression:
    ridge.score(train_feats,y)

    #Defining all the tresholds where to filter coefficients
    tresholds = np.linspace(np.min(ridge_coefs), np.max(ridge_coefs)/4, num=10)

    #Definition of:
    #the filter transformer
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
    x_train,x_test,y_train,y_test=tts(train_feats, y, test_size=0.1, shuffle=False)

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

    filename = pj(results_dir,'best_gpr_in_gridscearch.sav')
    #Saving the completly the best object after training
    joblib.dump(grid, filename)

    #Sending e-mail when finished
    username = 'alessandro.dagostino.notifica@gmail.com'
    password = 'notific@'
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(username,password)

    msg = MIMEMultipart()
    msg['From']=username
    msg['To']= 'alessandro.dagostino96@gmail.com'
    msg['Subject']='Training Finito'
    message = "Il training e' finito"

    msg.attach(MIMEText(message, 'plain'))
    server.send_message(msg)
    del msg

if __name__ == '__main__':
    main()
