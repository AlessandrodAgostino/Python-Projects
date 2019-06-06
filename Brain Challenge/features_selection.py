import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from os.path import join as pj
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge


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
        target = y=data['age_floor'].values
        return (data, target)

@dataclass
class ExcludeCategorical(BaseEstimator, TransformerMixin):
    #Return self nothing else to do here
    def fit( self, X, y = None ):
        return self

    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values

@dataclass
class FilterRidgeCoefficients(BaseEstimator, TransformerMixin):
    ridge_coefs : np.array

    def fit( self, X, y = None ):
        return self

    #Method that describes what we need this transformer to do
    def transform( self, X, y = None, tresh):
        filter =
        return X.loc[:,'lh_bankssts_area' :'rh.Whole_hippocampus'].values


Preparation = Pipeline([('selector', ExcludeCategorical()),
                        ('scaler', MinMaxScaler())])

data_train, y = LoadData('local')
data_train = Preparation.fit(data_train)

alphas=np.arange(0.001, 10, 0.005)
ridgecv=RidgeCV(alphas=alphas, fit_intercept=False)

#ADD THE SELECTION OF THE BEST RIDGE ALPHA

alpha=3.151
ridge=Ridge(alpha = alpha, fit_intercept=False)
