import numpy as np
import time
from scipy.ndimage import label
import matplotlib.pyplot as plt
from shapely.ops import polygonize,unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
from SALib.sample import saltelli
from skimage import io, morphology, img_as_uint, img_as_ubyte

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
image = io.imread('bw_nuclei_bound.png')
np.unique(image)
image.shape
image[:,:,1] = np.zeros(image[:,:,1].shape)
image[:,:,2] = np.zeros(image[:,:,2].shape)
image[:,:,3] = np.zeros(image[:,:,3].shape)

#%% Finding all the selections
# val = list(np.linspace(0,255,num=10))
# for min, max in zip(val, val[1:]):
#     sel_image = v_select(image[:,:,0], bot=min, top=max)
#     io.imsave('selections/{:.0f}_{:.2f}_selection.png'.format(min, max),
#               img_as_ubyte((sel_image>0).astype(float)))

#%%
#Working on nuclei
nuclei = io.imread('selections/0_28.33_selection.png')
np.unique(nuclei)
nuclei = morphology.binary_erosion(nuclei)
nuclei = morphology.binary_erosion(nuclei)
nuclei = morphology.binary_dilation(nuclei)
nuclei = morphology.binary_dilation(nuclei)
nuclei = nuclei*255
np.unique(nuclei)
io.imsave('nuclei.png'.format(min, max),img_as_ubyte((nuclei>0).astype(float)))

nuclei_lab, n_nuclei = label(nuclei, structure = s)
n_nuclei
image[:,:,1] = image[:,:,1] + nuclei_lab
#%%
#Finding all the cells (3% error)
cells = io.imread('selections/0_28.33_selection.png')
cells = cells - nuclei
cells = morphology.binary_dilation(cells)
cells = morphology.binary_dilation(cells)
cells = morphology.binary_dilation(cells)
cells = morphology.binary_dilation(cells)
cells = morphology.binary_dilation(cells)

cells = morphology.binary_erosion(cells)
cells = morphology.binary_erosion(cells)
cells = morphology.binary_erosion(cells)
cells = np.invert(cells)

io.imsave('cells.png',img_as_ubyte((cells>0).astype(float)))
cells_lab, n_cells = label(cells, structure = s)
n_cells
image[:,:,2] = image[:,:,2] + cells_lab
#%%
#Selecting lumes
lumes = io.imread('selections/142_170.00_selection.png')
lumes = morphology.binary_erosion(lumes)
lumes = morphology.binary_erosion(lumes)
lumes = morphology.binary_dilation(lumes)
lumes = morphology.binary_dilation(lumes)
lumes = morphology.binary_dilation(lumes)
lumes = morphology.binary_dilation(lumes)
lumes = morphology.binary_dilation(lumes)
lumes = morphology.binary_dilation(lumes)
lumes = morphology.binary_erosion(lumes)
lumes = morphology.binary_erosion(lumes)
lumes = morphology.binary_erosion(lumes)
lumes = morphology.binary_erosion(lumes)
lumes = lumes*255
io.imsave('lumes.png',img_as_ubyte((lumes>0).astype(float)))

lumes_lab, n_lumes = label(lumes, structure = s)
n_lumes
image[:,:,3] = image[:,:,3] + lumes_lab

#%%
#Selecting boundaries
bounds = io.imread('selections/57_85.00_selection.png')
bounds = morphology.binary_erosion(bounds)

bounds = morphology.binary_dilation(bounds)
bounds = morphology.binary_dilation(bounds)
bounds = morphology.binary_dilation(bounds)

bounds = bounds*255
io.imsave('bounds.png',img_as_ubyte((bounds>0).astype(float)))
bounds_lab, n_bounds = label(bounds, structure = s)
print(n_bounds)
image[:,:,3] = image[:,:,3] - bounds_lab
#%%

#-------------------------------------------------------------------------------









sel_epits = v_select(image[:,:,0], top = 30)
#sel_epits = morphology.binary_dilation(sel_epits)
epits, n_epits = label(sel_epits, structure = s)
#Saving labels
image[:,:,1] = image[:,:,1] + epits
io.imsave('big_epits.png', img_as_ubyte((epits>0).astype(float)))
#%%
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
io.imsave('big_lumes.png',img_as_ubyte((lumes>0).astype(float)))

#%%
#Selecting background
sel_back = v_select(image[:,:,0], bot = 201)
backg, _ = label(sel_back, structure = s)
io.imsave('big_backg.png',img_as_ubyte((backg>0).astype(float)))
#%%
#Testing correspondece between identified regions
np.unique(image[:,:,1])
epit1 = image[:,:,1] == 2
lume1 = image[:,:,1] == 256 -2
cell1 = epit1 + lume1
