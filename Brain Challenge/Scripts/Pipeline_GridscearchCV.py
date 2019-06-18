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
#from CachedFeaturesFilter_fit import CachedFeaturesFilter, location, memory
from sklearn.datasets import load_boston, load_iris
import warnings


location = './cachedir'
memory = Memory(location=location, verbose=1)
memory.clear(warn=False) #Command used to clear the memory

class CachedFeaturesFilter(BaseEstimator, TransformerMixin):
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

def main():
    #Remote path for Data and Results directories
    # data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
    # results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
    #
    # data_train = pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
    #                         header=0, sep='\t')
    # y = data_train['age_floor'].values
    # feats= data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values

    scaler = MinMaxScaler()
    scalers  = [MinMaxScaler(), StandardScaler()]
    alphas=np.linspace(0.001, 10, num=200)
    lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=1000)
    ridge=RidgeCV(alphas=alphas, fit_intercept=True)
    regressors = [lasso, ridge]

    GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=3, kernel=RBF())
    transformer = CachedFeaturesFilter(lasso, 0.01,True)


    pipeline = Pipeline([('Scaler', scaler),('Filter', transformer), ('GPR', GPR)])

    tresholds = np.linspace(0,0.25, num=100)
    parameter_grid = {'Scaler' : scalers,
                      'Filter__regressor' : regressors,
                      'Filter__treshold_mul' : tresholds,
                      'GPR__kernel' : [RBF(), DotProduct() + WhiteKernel()]}

    grid = GridSearchCV(pipeline, param_grid=parameter_grid, cv=2)

    X,y = load_boston(return_X_y=True)
    x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=False)

    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            grid.fit(x_train,y_train)
            end = time.time()
            print('\nThe grid fitting took {:.2f} s to compute.'.format(end - start))
    #In local it took 730s

    #transformer.history
    #plt.hist(transformer.history)
    grid.score(x_test,y_test)
    print("the best score was" ,grid.best_score_)

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

    #Saving the results
    history_df = pd.DataFrame(transformer.history)
    history_df.to_csv("/home/alessandro/Python/Brain Challenge/Results/history_df.csv",index = False, header = False)

    with open(pj('/home/alessandro/Python/Brain Challenge/Results/writing_results.txt'), 'w') as the_file:
        the_file.write('Result of the run of:\n')
        the_file.write(str(grid))
        the_file.write('\n\nBest parameters were:\n')
        the_file.write(str(grid.best_params_))
        the_file.write('\n\nBest score was\n')
        the_file.write(str(grid.best_score_))
        the_file.write('\n\nPrediction on the test set is:\n')
        the_file.write(str(grid.score(x_test,y_test)))
        the_file.write('\n')

    #filename = pj(results_dir,'best_params_in_gridscearch.pkl')
    filename = '/home/alessandro/Python/Brain Challenge/Results/best_params_in_gridscearch.pkl'
    joblib.dump(grid.best_params_,filename, compress=1)

    filename = '/home/alessandro/Python/Brain Challenge/Results/gridscearch.pkl'
    joblib.dump(grid,filename, compress=1)

if __name__ == '__main__':
    main()
