import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from os.path import join as pj
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as tts



#Load Data from server or from laptop
def LoadData(location):
        if location == "remote":
            data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
            results_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Results'
        elif location == "local":
            data_dir='/home/alessandro/Dropbox/UniBo/Brain Challenge/Data'
            results_dir='/home/alessandro/Dropbox/UniBo/Brain Challenge/Results'
        else: raise Exception('Location unknown')

        data = pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                                header=0, sep='\t')
        target = data['age_floor'].values
        return (data, target)

@dataclass
class ExcludeCategorical(BaseEstimator, TransformerMixin):
    #Return self nothing else to do here
    def fit( self, X, y = None ):
        return self

    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values


class FilterRidgeCoefficients(BaseEstimator, TransformerMixin):
    def __init__(self, ridge_coefs, treshold):
        self.ridge_coefs = ridge_coefs
        self.treshold = treshold

        #This array has to be given at the creation of the method

    def fit( self, X, y = None ):
        return self

    #The treshold is given as a variable parameter
    def transform( self, X, tresh=0.0, y = None):
        filter = ridge_coefs > tresh
        return X[filter]


Preparation = Pipeline([('selector', ExcludeCategorical()),
                        ('scaler', MinMaxScaler())])

data_train, y = LoadData('remote')
data_train = Preparation.fit_transform(data_train)

alphas=np.arange(0.001, 10, 0.005)
ridgecv=RidgeCV(alphas=alphas, fit_intercept=False)
ridgecv.fit(data_train,y)

alpha = ridgecv.alpha_
alpha
ridge_coefs = np.sort(np.abs(ridgecv.coef_))
len(ridge_coefs)
#ADD THE SELECTION OF THE BEST RIDGE ALPHA

#alpha=3.151
#ridge=Ridge(alpha = alpha, fit_intercept=False)

#Should use a true set of coeff
#ridge_coefs = np.linspace(2.0, 30.0, num=954)
best_ridge = FilterRidgeCoefficients(ridge_coefs)
tresholds = np.linspace(np.min(ridge_coefs), np.max(ridge_coefs), num=20)
tresholds


kernels = [DotProduct() + WhiteKernel(), RBF()]


GPR=GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=50, kernel=RBF())
GPR.fit(data_train, y)

x_train,x_test,y_train,y_test=tts(data_train, y, test_size=0.1, shuffle=False)
gpr_red_scores = []
for t in tresholds:
    red_train = FilterRidgeCoefficients(t).fit_transform(x_train)
    GPR.fit(red_train, y_train )
    spr_red_scores.append(GPR.score(red_train, y_train ))
    print(t)
