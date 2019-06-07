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
