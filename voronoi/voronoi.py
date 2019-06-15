import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def from_index_to_cord(region_index, voronoi):
    index_list = voronoi.regions[region_index]
    list_of_coord = np.zeros((len(index_list), 2))
    for n,i in enumerate(index_list):
        list_of_coord[n] = voronoi.points[i]
    return list_of_coord

def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

points2 = np.random.rand(30,2)
vor = Voronoi(points2)
vor_plot.savefig("vor_plot.png")

cord_region_0 = from_index_to_cord(0,vor)
#convex_0 = ConvexHull(cord_region_0)


points = np.array([(1, 2), (3, 4), (3, 6), (2, 4.5), (2.5, 5)])
hull = ConvexHull(cord_region_0)

np.random.seed(1)
random_points = np.random.uniform(0, 6, (100, 2))

for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1])

plt.scatter(*points.T, alpha=.5, color='k', s=200, marker='v')

for p in random_points:
    point_is_in_hull = point_in_hull(p, hull)
    marker = 'x' if point_is_in_hull else 'd'
    color = 'g' if point_is_in_hull else 'm'
    plt.scatter(p[0], p[1], marker=marker, color=color)

#%%
