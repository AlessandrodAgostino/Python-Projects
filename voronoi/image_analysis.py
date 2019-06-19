import numpy as np
from scipy.ndimage import label
from skimage import data, exposure
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt

filename = 'selection2.png'
file_write = 'gr_selection2.png'

selection = io.imread(filename)


gr_sel.size
hist, _ =exposure.histogram(gr_sel)
hist1, _ = np.histogram(gr_sel.ravel())
plt.hist(hist1)


io.imsave(file_write, gr_sel)
io.imshow(gr_sel)

labeled_image, num_features = label(gr_sel)
np.unique(gr_sel)
