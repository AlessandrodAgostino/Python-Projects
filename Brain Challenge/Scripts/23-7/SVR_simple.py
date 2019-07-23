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

x_train,x_test,y_train,y_test=tts(X, y, test_size=0.1, shuffle=True)

#%%
#Fitting and Saving
# pipes = []
# coefs = []
# times = []
#
#
# for sca, reg in product(scalers, regressors):
#     pipe = Pipeline([('scaler', sca), ('regressor', reg)])
#     st = time.time()
#     pipe.fit(x_train,y_train)
#     en = time.time()
#     print('\nThe fitting of {!s:.12} + {!s:.7} took {:.2f}s to compute'.format(pipe.named_steps['scaler'],pipe.named_steps['regressor'], en -st))
#     pipes.append(pipe)
#     coefs.append(pipe.named_steps['regressor'].coef_)
#     times.append(en-st)
#
# #Saving pipes items after fit
# for pipe in pipes:
#     filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
#     dump(pipe,pj(results_dir,'joblib',filename))

#%% Loading
pipes = []
coefs = []
for sca, reg in product(scalers, regressors):
    pipe = Pipeline([('scaler', sca), ('regressor', reg)])
    filename = "{!s:.5}_{!s:.5}.joblib".format(pipe.named_steps['scaler'],pipe.named_steps['regressor'])
    pipe = load(pj(results_dir,'joblib',filename))
    pipes.append(pipe)
    coefs.append(list(pipe.named_steps['regressor'].coef_))

#%% Plotting
# fig = plt.figure(figsize=(20, 20))
# for i,coef in enumerate(coefs):
#     plt.subplot(3, 3, i+1)
#     coef = np.sort(np.abs(coef))
#     plt.plot(coef[::-1], )
#     plt.axvline(x=50, color='r')
#     plt.axhline(y=coef[-50], color='g', label='50 feat threshold = {:.3}'.format(coef[-50]))
#     plt.legend(fontsize=12)
#     plt.title("Reg Coef for {!s:.12} + {!s:.10}".format(str(pipes[i][0]).split("(")[0],str(pipes[i][1]).split("CV")[0]),  fontsize = 15)
#     plt.tight_layout()
# fig.savefig(pj(results_dir,'NineCoefPlot.png'))

#%%
coefs = np.array(coefs)
ord_coefs = np.sort(np.abs(coefs),axis = 1)
feat_50 = ord_coefs[:,-50] #Values that correspond to the 50 feature treshold

filt0 = CoefFilter(feat_50[0], np.abs(coefs[0]))

SVR1 =SVR(kernel='linear', C=3)
cv=KF(10, shuffle=True)

pipe_filt = Pipeline([('scaler', scalers[0]), ('filt', filt0) ,('SVR', SVR1)])
pipe = Pipeline([('scaler', scalers[0]) ,('SVR', SVR1)])

list_par_grid_SVR = {'SVR__kernel': ["linear", 'poly', 'rbf', 'sigmoid'],\
                     'SVR__C': [5, 6.25, 7.5, 8.75, 10],\
                     'SVR__degree': [1, 2, 3, 4, 5, 6],\
                     'SVR__gamma': [0.1, 0.5,1,2.5,3]}

grid_SVR = GridSearchCV(pipe_filt,
                         param_grid = list_par_grid_SVR,
                         n_jobs=16,
                         pre_dispatch=8,
                         cv=cv)

grid_SVR.fit(x_train,y_train)
grid_SVR.score(x_test,y_test)
y_pred = grid_SVR.predict(x_test)

y_df = pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
y_df.to_csv(pj(results_dir,'y_df'), sep="\t")

filename="grid_SVR.joblib"
dump(grid_SVR, pj(results_dir,'joblib',filename))



#%% Plotting and regressing with GPR
#
#
# y_pred = pipe_filt.predict(x_test)
# y_pred.shape
# fig = plt.figure()
# plt.scatter(y_pred,y_test)
# plt.gca().set_title("Predictor's score: {:.2f}".format(pipe_filt.score(x_test,y_test)),
#                     size = 17)
# plt.gca().set_xlabel("y predicted on x_test", size=15)
# plt.gca().set_ylabel("y_test", size=15)
# fig.savefig(pj(results_dir, 'SVR_simple_2.png'), bbox_inches='tight')
# y_int = np.linspace(18,80,100)
#
# GPR_y = GaussianProcessRegressor(n_restarts_optimizer=50, normalize_y=True, kernel=Matern())
# GPR_y.fit(y_pred.reshape(-1, 1),y_test)
# GPR_y.predict(y_int.reshape(1, -1))
