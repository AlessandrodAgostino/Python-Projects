import numpy as np
import time

from scipy.ndimage import label

import matplotlib.pyplot as plt
from shapely.ops import polygonize,unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
from SALib.sample import saltelli
from skimage import io


thetas = np.linspace(0,2*np.pi,15)
r = 1
c = (2,3)
x = np.cos(thetas)*r + c[0]
y = np.sin(thetas)*r + c[1]
circle_point = np.stack((x,y)).T
circle = Polygon(circle_point)

thetas = np.linspace(0,2*np.pi,15)
r = 0.5
c = (2,3)
x = np.cos(thetas)*r + c[0]
y = np.sin(thetas)*r + c[1]
internal_point = np.stack((x,y)).T
internal = Polygon(internal_point)



fig = plt.figure()
plt.fill(*circle.exterior.xy, 'k', alpha = 0.5)
plt.fill(*internal.exterior.xy, 'k')

plt.ylim(0,5)
plt.xlim(0,5)
plt.axis('off')
fig.savefig('small_black_circle.png', bbox_inches='tight',dpi=10)

#%%
from scipy.ndimage import label, generate_binary_structure
a = np.array([[0,0,0,0,0,0,0,0],
              [0,0,0,1,1,0,0,0],
              [0,0,2,0,0,1,0,0],
              [0,0,1,0,0,1,0,0],
              [0,0,0,1,3,0,0,0],
              [0,0,0,0,0,0,0,0]])
s = [[1,1,1],
     [1,1,1],
     [1,1,1]]
labeled_array, num_features = label(a, structure = s)
labeled_array

#%%
image = io.imread('small_black_circle.png')
image = image[:,:,0]
image.shape
image.size

s = [[1,1,1],
     [1,1,1],
     [1,1,1]]
#%%
def select_dark(pix):
    if pix <= 60: return 1
    else: return 0

v_select_dark = np.vectorize(select_dark)
sel_dark = v_select_dark(image)

core, _ = label(sel_dark, structure = s)
io.imsave('core.png',core*255)
#%%
def select_grey(pix):
    if pix >= 60 and pix <= 220: return 1
    else: return 0

v_select_grey = np.vectorize(select_grey)
sel_grey = v_select_grey(image)
contour, _ = label(sel_grey, structure = s)
io.imsave('contour.png',contour*255)
#%%
def select_white(pix):
    if pix >= 220: return 1
    else: return 0

v_select_white = np.vectorize(select_white)
sel_white = v_select_white(image)
backg, _ = label(sel_white, structure = s)
io.imsave('backgroung.png',backg*255)
