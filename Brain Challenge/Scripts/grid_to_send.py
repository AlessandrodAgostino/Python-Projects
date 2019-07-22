#Alessandro d'dagostino

import numpy as np
import pylab as plt
import pandas as pd
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
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold as KF
from joblib import dump, load
from os.path import join as pj


class CoefFilter(BaseEstimator, TransformerMixin):
    """
    This is the customed Filter, it's a new class inheriting from two class of SKlearn.
    It simply select the columns tht received a coefficient greater than a certain value in the regression.
    The set of coefficients and the threshold have to be given as parameters during the initialization.
    """


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


def main():

    #Data directory on server
    data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
    results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
    #Uploading the data in a pandas dataframe
    data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                            header=0, sep='\t')
    #Selecting numerical features from the df. `.values` load the data as a Numpy array.
    feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
    #Selecting  the target variable
    y = data_train['age_floor'].values
    X = feats #n_features = 954 n_samples = 2364
#%%
#-------------------------------------------------------------------------------
    """
    Here I create all the parameters necessary to define the three scaling methods
    and the three regression method to be used in combination.
    """
    alphas=np.arange(0.001, 10, 0.005)
    scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()] #list of scalers
    lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
    ridge = RidgeCV(alphas=alphas, cv=5)
    elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
    regressors = [lasso, ridge, elnet] #list of regressors

    #Splitting the data in train set and test set.
    x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=True)


#%%
#-------------------------------------------------------------------------------
    """
    Here I fit on the data all the combinations of scalers and regressors and
    save the results using `joblib` in order to spare time in following executions.
    The fitting takes place using a pipeline with two steps.
    """

    #Lists that will contain the results
    pipes = []
    coefs = []
    times = []

    for sca, reg in product(scalers, regressors):
        pipe = Pipeline([('scaler', sca), ('regressor', reg)]) #def of pipeline
        X = x_train
        st = time.time()
        pipe.fit(X,y_train) #fitting
        en = time.time()

        #Appending the results
        pipes.append(pipe)
        coefs.append(pipe.named_steps['regressor'].coef_)
        times.append(en-st)
#%%
#-------------------------------------------------------------------------------
    """
    Saving the coefficients
    """
    for pipe in pipes:
        filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
        dump(pipe,pj(results_dir,filename))
#%%
#-------------------------------------------------------------------------------
    """
    Here I load the previous results.
    If you had already made the run of previous block you can just execute the following instead.
    """

    pipes = [] #List of fitted pipelines
    coefs = [] #List of pipeline's regressor's coefficients
    for sca, reg in product(scalers, regressors):
        pipe = Pipeline([('scaler', sca), ('regressor', reg)])
        filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
        pipe = load(pj(results_dir,filename))
        pipes.append(pipe)
        coefs.append(list(pipe.named_steps['regressor'].coef_))

#%%
#-------------------------------------------------------------------------------
    """
    Raw plotting of the different sets of coefficients.
    """

    fig = plt.figure(figsize=(20, 20))
    for i,coef in enumerate(coefs):#<--- limitation
        plt.subplot(3, 3, i+1)
        coef = np.sort(np.abs(coef))
        plt.plot(coef[::-1], )
        plt.axvline(x=50, color='r')
        plt.axhline(y=coef[-50], color='g', label='50 feat threshold = {:.3}'.format(coef[-50]))
        plt.legend(fontsize=12)
        plt.title("Reg Coef for {!s:.12} + {!s:.10}".format(str(pipes[i][0]).split("(")[0],str(pipes[i][1]).split("CV")[0]),  fontsize = 15)
        plt.tight_layout()
    fig.savefig(pj(results_dir,'NineCoefPlot.png'))

#%%
#-------------------------------------------------------------------------------
    """
    Here I start the analysis.
    """

    ord_coefs = np.sort(np.abs(np.array(coefs)),axis = 1)
    feat_50 = ord_coefs[:,-50] #Values of the 50th highest coefficients


    #Resuming the single steps of each pipeline
    scals = [pipes[n].named_steps['scaler'] for n in range(9)]
    filts = [CoefFilter(feat_50[n], ord_coefs[n]) for n in range(9)]
    #Number of different cuts to make on the features.
    n_tresh = 10
    #Thresholds values to make those cuts
    treshs = [np.linspace(feat_50[n], ord_coefs[n,-1], num=n_tresh) for n in range(9)]

    """
    #If you want to train a SVR, this could be an exemple of the code:

    list_par_grid_SVR = [{'Scaler': [scals[n]],\
                          'Filter': [filts[n]],\
                          'Filter__coef':[ord_coefs[n]],\
                          'SVR__kernel': ["linear", 'poly', 'rbf', 'sigmoid'],\
                          'SVR__C': [0.1, 0.5, 1, 5, 10],\
                          'SVR__degree': [1, 2, 3, 4, 5, 6],\
                          'SVR__gamma': [0.001, 0.01, 0.1, 1, 'auto']} for n in range(9)]
    SVR1 =SVR(kernel='linear', C=3)

    cv=KF(10, shuffle=True)
    pipe = Pipeline([('Scaler', scals[0]), ('Filter', filts[0]), ('SVR', SVR1)])

    grid = GridSearchCV(pipe, param_grid = list_par_grid_SVR, n_jobs=16, pre_dispatch=8,  cv=cv)
    """


    #If you want to train a GPR, this could be an exemple of the code:
    list_par_grid_same_kernel = [{'Scaler': [scals[n]], 'Filter': [filts[n]],'Filter__coef':[ord_coefs[n]],'Filter__treshold': treshs[n]} for n in range(9)]
    GPR = GaussianProcessRegressor(n_restarts_optimizer=50, normalize_y=True, kernel=Matern())
    cv=KF(10, shuffle=True)
    pipe = Pipeline([('Scaler', scals[0]), ('Filter', filts[0]), ('GPR', GPR)])
    grid = GridSearchCV(pipe, param_grid = list_par_grid_same_kernel, n_jobs=16, pre_dispatch=8,  cv=cv)


    #Here there is the actual fitting. This command takes a lot of time to be computed (~ hours on bio10)
    st = time.time()
    grid.fit(x_train,y_train)
    en = time.time()
    print("\nThe fit took {:.2f}s".format(en-st))
#%%
#-------------------------------------------------------------------------------
    """
    Here I save all the results.
    """

    #Saving all the grid found by the gridscearch
    filename = "gridscearch.pkl"
    dump(grid,pj(results_dir, filename))
#%%
#-------------------------------------------------------------------------------
    """
    Here I load all the results.
    """

    filename = "gridscearch.pkl"
    loaded_grid = load(pj(results_dir, filename))
#%%
#-------------------------------------------------------------------------------
    """
    Here I load all the results.
    """

    #This is the best estimator found
    best_pipe = loaded_grid.best_estimator_

#%%
#-------------------------------------------------------------------------------
    """
    Fitting the the test set using the best estimator.
    """

    best_pipe.fit(x_test,y_test)
    # best_pipe.score(x_test,y_test)

    filename = "best_pipe_on_test.pkl"
    dump(best_pipe,pj(results_dir, filename))

    filename = "y_test.pkl"
    dump(y_test,pj(results_dir, filename))

if __name__ == '__main__':
    main()
