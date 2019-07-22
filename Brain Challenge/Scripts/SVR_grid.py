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
#Sending emails
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class CoefFilter(BaseEstimator, TransformerMixin):
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

#Function to send e-mail to monitoring the state of the fit
def send_email(message):
    username = 'alessandro.dagostino.notifica@gmail.com'
    password = 'notific@'
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(username,password)

    msg = MIMEMultipart()
    msg['From']=username
    msg['To']= 'alessandro.dagostino96@gmail.com'
    msg['Subject']="Training - Redoing SVR"

    msg.attach(MIMEText(message, 'plain'))
    server.send_message(msg)
    del msg

def main():

    try:
        send_email("Start of the running")
        #Defining all the data
        data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
        results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
        data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                                header=0, sep='\t')
        feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
        y = data_train['age_floor'].values
        X = feats #n_features = 954 n_samples = 2364

        #Defining all the scalers and regressors necessary for the coefficient scearching
        alphas=np.arange(0.001, 10, 0.005)
        scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
        lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
        ridge = RidgeCV(alphas=alphas, cv=5)
        elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
        regressors = [lasso, ridge, elnet]

        send_email("Loading the coefficients in \n" + results_dir)

        #Loading from joblib files pipes item already fit and their coefficients separately
        pipes = []
        coefs = []
        for sca, reg in product(scalers, regressors):
            pipe = Pipeline([('scaler', sca), ('regressor', reg)])
            filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
            pipe = load(pj(results_dir,filename))
            pipes.append(pipe)
            coefs.append(list(pipe.named_steps['regressor'].coef_))

        ord_coefs = np.sort(np.abs(np.array(coefs)),axis = 1)
        feat_50 = ord_coefs[:,-50] #Values that correspond to the 50 feature treshold

        send_email("Starting the fit on all the nine combinations with: \n\t [RBF(), Matern(), RationalQuadratic(),DotProduct() + WhiteKernel()]")

        x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=True)
        scals = [pipes[n].named_steps['scaler'] for n in range(9)]
        filts = [CoefFilter(feat_50[n], ord_coefs[n]) for n in range(9)]
        n_tresh = 10
        treshs = [np.linspace(feat_50[n], ord_coefs[n,-1], num=n_tresh) for n in range(9)]

        #with different kernels but always on 50 features
        list_par_grid_SVR = [{'Filter': [filts[n]],'Filter__coef':[ord_coefs[n]],'SVR__kernel': ["linear", 'poly', 'rbf', 'sigmoid'],'SVR__C': [5, 6.25, 7.5, 8.75, 10],'SVR__degree': [1, 2, 3, 4, 5, 6],'SVR__gamma': [0.1, 0.5,1,2.5,3]} for n in range(3)]


        scaled_x_train = scals[0].fit_transform(x_train)

        SVR1 =SVR(kernel='linear', C=3)
        cv=KF(10, shuffle=True)

        pipe = Pipeline([('Filter', filts[0]), ('SVR', SVR1)])

        one_kernel_grid = GridSearchCV(pipe, param_grid = list_par_grid_SVR, n_jobs=16, pre_dispatch=8,  cv=cv)
        st = time.time()
        one_kernel_grid.fit(scaled_x_train,y_train)
        en = time.time()
        print("\nThe fit took {:.2f}s".format(en-st))

        send_email("End of the fit."+"\nIt took {:.2f}s".format(en-st))

        #Saving all the grid found by the gridscearch
        filename = "brain_SVR_gridscearch_redo.pkl"
        dump(one_kernel_grid,pj(results_dir, filename))
        #
        send_email("Everything has been saved")

        filename = "brain_SVR_gridscearch_redo.pkl"
        loaded_grid = load(pj(results_dir, filename))

        #%%
        best_pipe = loaded_grid.best_estimator_

        scaled_x_test = scals[0].fit_transform(x_test)

        print("The best pipe was: ", best_pipe)

        best_pipe.fit(scaled_x_train,y_train)
        best_pipe.score(scaled_x_test,y_test)
        #0.7245557808714778


        #%%
        fig = plt.figure()
        y_pred_on_test = best_pipe.predict(scaled_x_test)
        plt.scatter(y_test,y_pred_on_test)
        plt.gca().set_title("Predictor's score: {:.2f}".format(best_pipe.score(x_test,y_test)),
                            size = 17)
        plt.gca().set_ylabel("y predicted on x_test", size=15)
        plt.gca().set_xlabel("y_test", size=15)
        fig.savefig(pj(results_dir, 'After_train_SVR_redo.png'), bbox_inches='tight')


    except ValueError:
        send_email("Something went wrong! Exception Raised!")


if __name__ == '__main__':
    main()
