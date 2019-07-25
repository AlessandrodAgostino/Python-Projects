import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pj
from joblib import dump, load


data_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Data'
scripts_dir='/home/STUDENTI/alessandr.dagostino2/Python-Projects/Brain Challenge/Scripts'

# data_dir='/home/alessandro/Python/Brain Challenge/Data'
# scripts_dir = '/home/alessandro/Python/Brain Challenge/Scripts'

data_train=pd.read_csv(pj(data_dir, 'Training_Set_YESregressBYeTIVifCorr_LogScaled_combat_SVA.txt'),
                       header=0, sep='\t')
id_features = data_train.loc[:,[i for i in data_train.columns if 'age' in i or 'gender' in i or 'site' in i]]

#%%
ax = sns.violinplot(x="site", y="age", hue="gender",
                    data=id_features.loc[id_features["site"]<=8], palette="muted", split=True, inner = None)
ax = sns.swarmplot(x="site", y="age", data=id_features.loc[id_features["site"]<=8],
                   color="black", size=3)

ax.get_figure().savefig(pj(scripts_dir,'24-7','violin_swarm_gender_site_1_8.png'), dpi=1000)
#%%
ax = sns.violinplot(x="site", y="age", hue="gender",
                    data=id_features.loc[id_features["site"]>8], palette="muted", split=True, inner = None)
ax = sns.swarmplot(x="site", y="age", data=id_features.loc[id_features["site"]>8],
                   color="black", size=3)
ax.get_figure().savefig(pj(scripts_dir,'24-7','violin_swarm_gender_site_9_16.png'), dpi=1000)

#%%
ax = sns.violinplot(x="site", y="age", hue="gender",
                    data=id_features, palette="muted", split=True)
ax.get_figure().savefig(pj(scripts_dir,'24-7','violin_gender_site.png'), dpi=1000)
#%%
g = sns.FacetGrid(id_features, hue="gender")
g =(g.map(sns.distplot, "age")
    .add_legend())
g.savefig(pj(scripts_dir,'24-7','age_dist_gender.png'), dpi=1000)

#%%
g = sns.FacetGrid(id_features, hue="site")
g =(g.map(sns.distplot, "age")
    .add_legend())
g.savefig(pj(scripts_dir,'24-7','age_dist_site.png'), dpi=1000)
