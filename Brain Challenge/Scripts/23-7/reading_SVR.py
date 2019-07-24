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

#%%
data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                        header=0, sep='\t')
feats = data_train.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values
y = data_train['age_floor'].values
X = feats #n_features = 954 n_samples = 2364

scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
alphas=np.arange(0.001, 10, 0.005)
lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5)
ridge = RidgeCV(alphas=alphas, cv=5)
elnet = ElasticNetCV(alphas=alphas, max_iter=100000, cv=5)
regressors = [lasso, ridge, elnet]

#%% Loading
pipes = []
coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(pj(results_dir,'joblib',filename))
    pipes.append(pipe)
    coefs.append(list(pipe.named_steps['regressor'].coef_))

y_df=pd.read_csv(pj(results_dir,'y_df'), sep='\t')


filename="grid_SVR.joblib"
grid_SVR = load(pj(results_dir,'joblib',filename))


grid_SVR.best_score_
y_df.head()

y_test = y_df['y_test'].values
y_pred = y_df['y_pred'].values

fig = plt.figure()
plt.scatter(y_test,y_pred)
plt.gca().set_title("Predictor's score: {:.3f}".format(grid_SVR.best_score_, size = 17))
plt.gca().set_ylabel("Predicted age", size=15)
plt.gca().set_xlabel("Age", size=15)
fig.savefig(pj(results_dir, '24-7_SVR.png'), bbox_inches='tight')

#%%
GPR_y = GaussianProcessRegressor(normalize_y=True, kernel=Matern())
GPR_y.fit(y_pred.reshape(-1,1),y_test)
y_int = np.linspace(np.min(y_test), np.max(y_test), num = 100)
GPR_y.predict(y_int.reshape(-1, 1))
