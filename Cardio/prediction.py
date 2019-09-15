import numpy as np
import pylab as plt
import pandas as pd
from itertools import product
import seaborn as sns
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
from sklearn.metrics import r2_score



#%%
cardio_dir='/home/STUDENTI/alessandr.dagostino2/Cardio'
data = load(pj(cardio_dir,"dati.pickle"))

not_num_features = ["age","AA", "RR", "Rpeakvalues",
                    "rhythm", "sex","signal","time",
                    "mean_time","mean_signal","var_mean_signal"]

features = data.loc[:,[i for i in data.columns if i not in not_num_features]]
features.apply(lambda col:pd.to_numeric(col, errors='coerce'))
X = features.values
y = data["age"].values

#%%
y_pred =pd.read_csv(pj('/home/STUDENTI/alessandr.dagostino2/Python-Projects/Cardio/pred_20_10_5.csv'), sep='\t')

y_pred.head()
melt_y_df = pd.melt(y_pred,
                    id_vars=['y_test'],
                    value_vars=['y_pred_all', 'y_pred_20', 'y_pred_10', 'y_pred_5'],
                    var_name='run',
                    value_name = 'y_pred')

#%%
r2 = []
r2.append(r2_score(y_pred['y_test'], y_pred['y_pred_all']))
r2.append(r2_score(y_pred['y_test'], y_pred['y_pred_20']))
r2.append(r2_score(y_pred['y_test'], y_pred['y_pred_10']))
r2.append(r2_score(y_pred['y_test'], y_pred['y_pred_5']))

r2

#%%
lp = sns.lmplot(data=melt_y_df,
                x='y_test',
                y='y_pred',
                hue='run',
                col='run',
                robust=True,
                legend=True,\
                scatter_kws = dict(alpha = 0.5))
fig = lp.fig
for n,ax in enumerate(fig.axes):
    ax.set_title("run = {}  $R^2=${:.4f}".format(y_pred.columns[-len(fig.axes)+n], r2[n] ))
fig.savefig(pj('/home/STUDENTI/alessandr.dagostino2/Python-Projects/Cardio/','four_graphs.png'))
