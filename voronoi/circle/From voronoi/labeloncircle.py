import numpy as np
import time
from scipy.ndimage import label
import matplotlib.pyplot as plt
from shapely.ops import polygonize,unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
from SALib.sample import saltelli
from skimage import io, morphology, img_as_uint

#structure elements for recognizing features
s = [[1,1,1],
     [1,1,1],
     [1,1,1]]

def select(pix, bot = 0, top = 255):
    if pix >= bot and pix <= top: return 1
    else: return 0
v_select = np.vectorize(select)

#%%
####BIG IMAGE
image = io.imread('7995_regions_bound_nuclei.png')
image.shape
image = image[:,:,1:3]
image[:,:,1] = np.zeros(image[:,:,1].shape)

#Selecting lumes
sel_epits = v_select(image[:,:,0], top = 30)
sel_epits = morphology.binary_dilation(sel_epits)
epits, n_epits = label(sel_epits, structure = s)
#Saving labels
image[:,:,1] = image[:,:,1] + epits
io.imsave('big_epits.png', img_as_uint((epits>0).astype(float)))

#Selecting epitelia
sel_lumes = v_select(image[:,:,0], bot = 31, top = 200)
lumes, n_lumes = label(sel_lumes, structure = s)
n_lumes
n_erosion = 0
n_epits

while n_lumes > n_epits:
    print("erosion")
    sel_lumes = morphology.binary_erosion(sel_lumes)
    lumes, n_lumes = label(sel_lumes, structure = s)
    n_erosion =+ n_erosion

for _ in range(n_erosion):
    print("dilation")
    sel_lumes = morphology.binary_dilation(sel_lumes)

#Saving labels
image[:,:,1] = image[:,:,1] - lumes
io.imsave('big_lumes.png',img_as_uint((lumes>0).astype(float)))

#%%
#Selecting background
sel_back = v_select(image[:,:,0], bot = 201)
backg, _ = label(sel_back, structure = s)
io.imsave('big_backg.png',img_as_uint((backg>0).astype(float)))
#%%
#Testing correspondece between identified regions
np.unique(image[:,:,1])
epit1 = image[:,:,1] == 2
lume1 = image[:,:,1] == 256 -2
cell1 = epit1 + lume1
