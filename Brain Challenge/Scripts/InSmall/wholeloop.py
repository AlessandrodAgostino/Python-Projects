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
from sklearn.datasets import load_boston, load_iris
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold as KF
from joblib import dump, load
from os.path import join as pj
#Sending emails
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
    msg['Subject']="Training - InSmall 5"

    msg.attach(MIMEText(message, 'plain'))
    server.send_message(msg)
    del msg


def main():
    data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
    results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results/InSmallResults'
    data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                            header=0, sep='\t')
    feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
    y = data_train['age_floor'].values
    X = feats #n_features = 954 n_samples = 2364
    x_train,x_test,y_train,y_test=tts(X, y, test_size=0.3, shuffle=False)

    #Defining all the scalers and regressors necessary for the coefficient scearching
    alphas=np.arange(0.001, 10, 0.005)
    scaler = MinMaxScaler()
    lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
    ridge = RidgeCV(alphas=alphas, cv=5)
    regressor = ridge

    x_train_tran = scaler.fit_transform(x_train)
    regressor.fit(x_train_tran,y_train)
    x_test_tran = scaler.fit_transform(x_test)
    regressor.score(x_test_tran, y_test)


    coef = regressor.coef_
    coef = np.abs(coef)
    sort_coef = np.sort(coef)
    feat_50 = sort_coef[-50]
    filt = np.abs(coef) >= feat_50
    x_train_tran_filt = x_train_tran[:,filt]
    GPR = GaussianProcessRegressor(n_restarts_optimizer=50, kernel=Matern())
    GPR.fit(x_train_tran_filt,y_train)

    x_test_tran = scaler.fit_transform(x_test)
    x_test_tran_filt = x_test_tran[:,filt]
    GPR.score(x_test_tran_filt, y_test)

#%% 25 feat
    feat_25 = sort_coef[-25]
    filt_25 = coef >= feat_25
    x_train_tran_filt = x_train_tran[:,filt_25]
    GPR_25 = GaussianProcessRegressor(n_restarts_optimizer=50, kernel=Matern())
    GPR_25.fit(x_train_tran_filt,y_train)

    x_test_tran = scaler.fit_transform(x_test)
    x_test_tran_filt = x_test_tran[:,filt_25]
    GPR_25.score(x_test_tran_filt, y_test)
#%% 10 feats
    feat_10 = sort_coef[-10]
    filt_10 = coef >= feat_10
    x_train_tran = scaler.fit_transform(x_train)
    x_train_tran_filt = x_train_tran[:,filt_10]
    x_train_tran_filt
    GPR_10 = GaussianProcessRegressor(n_restarts_optimizer=50, kernel=Matern())
    GPR_10.fit(x_train_tran_filt,y_train)
    x_test_tran = scaler.fit_transform(x_test)
    x_test_tran_filt = x_test_tran[:,filt_10]
    GPR_10.score(x_test_tran_filt, y_test)


if __name__ == '__main__':
    main()
