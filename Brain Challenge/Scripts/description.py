import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pj
from joblib import dump, load


data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
scripts_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Scripts'

#Loading Data
data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                       header=0, sep='\t')

#Extracting descriptive features
id_features = data_train.loc[:,[i for i in data_train.columns if 'age' in i or 'gender' in i or 'site' in i]]

#Drawing the desired plots
ax = sns.violinplot(x="site", y="age", hue="gender",
                    data=id_features, palette="muted", split=True)
ax.get_figure().savefig(pj(scripts_dir,'violin_gender_site.png'), dpi=1000)
