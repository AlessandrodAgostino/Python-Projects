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
selection = io.imread('small_black_circle.png')
selection = selection[:,:,0]
selection.shape
selection.size
plt.hist(selection)

selection > 150



selection[selection > 150]=255
selection[selection > 90]=128
selection[selection < 90]=0
np.unique(selection)
plt.hist(selection)

labeled_image, num_features = label(selection)
num_features
np.unique(labeled_image)
