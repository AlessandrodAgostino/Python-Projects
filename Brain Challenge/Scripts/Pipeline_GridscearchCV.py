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
import joblib
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
    The treshold is inteded as a multiplier t such:
        min_coef = np.min(self.regr_coef)
        delta = np.max(self.regr_coef) - min_coef
        filter = self.regr_coef > min_coef + delta*self.treshold_mul

    It exploit the cache memory in order to not recompute those fitting that has
    already been done.
    It requires that the passed regressor has an attribute `coef_` in which the
    coefficients are stored.
    """
    def __init__(self, scaler ,regressor, treshold_mul):
        self.scaler = scaler #The way to scale datasets
        self.regressor = regressor #The way to regress data
        self.treshold_mul = treshold_mul #Where to cut the coefficients
        self.regr_coef = 0

    def fit( self, X, y = None ):
        cached_pipe = Pipeline([('scaler', self.scaler),
                                ('regressor', self.regressor)])
        cached_fit = memory.cache(cached_pipe.fit)
        cached_fit(X,y)
        if hasattr(cached_pipe[1], 'coef_'):
            self.regr_coef = cached_pipe[1].coef_
        else:
            self.regr_coef = np.random.rand(len(X[0,:])) #required attribute
        self.regr_coef = np.sort(np.abs(self.regr_coef))
        return self

    def transform( self, X, y = None):
        min_coef = np.min(self.regr_coef)
        delta = np.max(self.regr_coef) - min_coef
        filter = self.regr_coef > min_coef + delta*self.treshold_mul
        return X[:,filter]


def main():
    #Definition of my custom Tranformer
    location = './.cachedir'
    memory = Memory(location=location, verbose=1)
    memory.clear(warn=False)

    #Remote path for Data and Results directories
    data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
    results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'

    data_train = pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                            header=0, sep='\t')
    y = data_train['age_floor'].values
    feats= data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values

    scaler = MinMaxScaler()
    alphas=np.arange(0.001, 10, 0.005)
    lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=100)
    ridge=RidgeCV(alphas=alphas, fit_intercept=True, max_iter=100)
    regressors = [lasso, ridge]
    transformer = CachedFeaturesFilter(scaler, lasso, 1)

    GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())

    pipeline = Pipeline([('Filter', transformer), ('GPR', GPR)])

    tresholds = np.linspace(0.0,0.25, num=1000)
    parameter_grid = {'Filter__regressor' : regressors,
                      'Filter__treshold_mul' : tresholds,
                      'GPR__kernel' : [RBF(), DotProduct() + WhiteKernel()]}
    grid = GridSearchCV(pipeline, param_grid=parameter_grid, cv=2)

    x_train,x_test,y_train,y_test=tts(feats, y, test_size=0.1, shuffle=False)
    start = time.time()
    grid.fit(x_train,y_train)
    end = time.time()
    print('\nThe grid fitting took {:.2f} s to compute.'.format(end - start))

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


    filename = pj(results_dir,'best_par_in_gridscearch.pkl')
    joblib.dump(grid.best_params_,filename, compress=1)

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
