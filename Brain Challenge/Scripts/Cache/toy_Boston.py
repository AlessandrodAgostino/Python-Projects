import numpy as np
import pandas as pd
import seaborn as sns
import time
import pylab as plt
#import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from sklearn.datasets import load_boston
#Import the customed filter

location = './.cachedir'
memory = Memory(location=location, verbose=1)
memory.clear(warn=False)

class CachedFeaturesFilter(BaseEstimator, TransformerMixin):
    """docstring for FilterRidgeCoefficients
    This custom transform filter those
    features with a weight greater than a certain treshold.
    The treshold is inteded as a multiplier `t` such:
        min_coef = np.min(self.regr_coef)
        delta = np.max(self.regr_coef) - min_coef
        filter = self.regr_coef > min_coef + delta*self.treshold_mul

    It exploit the cache memory in order to not recompute those fitting that has
    already been done.
    It requires that the passed regressor has an attribute `coef_` in which the
    coefficients are stored.
    The attribute `history` is inteded to be a list.
    """
    def __init__(self, scaler ,regressor, treshold_mul , selection_history = False, history = None):
        self.scaler = scaler #The way to scale datasets
        self.regressor = regressor #The way to regress data
        self.treshold_mul = treshold_mul #Where to cut the coefficients
        self.regr_coef = 0
        self.selection_history = selection_history
        self.history = history

    def fit( self, X, y = None ):
        #Here I define a local pipeline that will be cached
        cached_pipe = Pipeline([('scaler', self.scaler),
                                ('regressor', self.regressor)])
        #Here I cache the fit method of the local pipeline, that will be called below
        cached_fit = memory.cache(cached_pipe.fit)
        cached_fit(X,y)
        if hasattr(cached_pipe.steps[1], 'coef_'):
            self.regr_coef = cached_pipe.steps[1].coef_
        else:
            self.regr_coef = np.random.rand(len(X[0,:])) #required attribute
        #Here I save the regression coefficients in absolut value
        self.regr_coef = np.abs(self.regr_coef)
        return self

    def transform( self, X, y = None):
        #Here I seek the Min and the Max of the coefficients and set the treshold
        min_coef = np.min(self.regr_coef)
        delta = np.max(self.regr_coef) - min_coef
        filter = self.regr_coef > min_coef + delta*self.treshold_mul
        #Saving the history of filtering
        if self.selection_history: self.history.append(filter*1)
        return X[:,filter]


def main():
    #Uploading toy DataFrame
    X, y = load_boston(return_X_y=True)

    scaler = MinMaxScaler()
    alphas=np.arange(0.001, 10, 0.005)
    lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=100)
    ridge=RidgeCV(alphas=alphas, fit_intercept=True)

    regressors = [lasso, ridge]
    scalers = [MinMaxScaler(), StandardScaler()]

    history3=[]

    transformer = CachedFeaturesFilter(scaler, lasso, 0.5, selection_history = True, history = history3)
    GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())

    pipeline = Pipeline([('Filter', transformer),
                         ('GPR', GPR)])

    tresholds = np.linspace(0,0.25, num=4)
    parameter_grid = {'Filter__scaler' : scalers,
                      'Filter__regressor' : regressors,
                      'Filter__treshold_mul' : tresholds,
                      'GPR__kernel' : [RBF(), DotProduct() + WhiteKernel()]}

    red_parameter_grid = {'Filter__treshold_mul' : tresholds}

    grid = GridSearchCV(pipeline, param_grid=red_parameter_grid, cv=2)

    x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=False)

    start = time.time()
    grid.fit(x_train,y_train)
    end = time.time()
    print('\nThe grid fitting took {:.2f} s to compute.'.format(end - start))
    history3

    #Saving the history of filter on the feautures
    history_df = pd.DataFrame(history3)
    history_df.to_csv("history_df.csv",index = False, header = False)

    #Saving the best parameters found by the grid
    filename = pj('toy_Boston_best.pkl')
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
    msg['Subject']='Training Boston Finito'
    message = "Il training su Boston e' finito"

    msg.attach(MIMEText(message, 'plain'))
    server.send_message(msg)
    del msg

if __name__ == '__main__':
    main()
