import numpy as np
import time

import matplotlib.pyplot as plt
from shapely.ops import polygonize,unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
from SALib.sample import saltelli
#scipy.ndimage.labelidentity

#Low discrepancy sampling of the plane:
problem = {'num_vars': 2,
           'names': ['x', 'y'],
           'bounds': [[0, 100],[0, 100],]}

start = time.time()
low_points = saltelli.sample(problem, 5000)
end = time.time()
print('\nThe time for sampling is {:.2f} s.'.format(end - start))
#0.12s

start = time.time()
vor = Voronoi(low_points)
end = time.time()
print('\nThe time for computing the Voronoi tassellation is {:.2f} s.'.format(end - start))
#0.40s

start = time.time()
#CUTTING the global Voronoi tassellation to avoid divergences
lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line ]
convex_hull = MultiPoint([Point(i) for i in low_points]).convex_hull.buffer(2)
tassel = MultiPolygon([poly.intersection(convex_hull) for poly in polygonize(lines)])
tassel = MultiPolygon([p for p in tassel] + [p for p in convex_hull.difference(unary_union(tassel))])
end = time.time()
print('\nThe time for clipping the tassellation is {:.2f} s.'.format(end - start))
#8.95 s

#Dictionary with the identity of all the circles in the image
circles_id = [{'r' : 10, 'c' : (40,30)},
              {'r' : 5, 'c' : (70,90)},
              {'r' : 15, 'c' : (20,60)}]

#List of the corrispondent polygons
circles = []
thetas = np.linspace(0,2*np.pi,15)
for c_id in circles_id:
    x = np.cos(thetas)*c_id['r'] + c_id['c'][0]
    y = np.sin(thetas)*c_id['r'] + c_id['c'][1]
    circle_point = np.stack((x,y)).T
    circles.append(Polygon(circle_point))

#Return the color of the region depending on whether it intersects the boundary, the interior or not intersects the circle
def color(r,circles):
    for circle in circles:
        if r.intersects(circle.boundary): return "r"
        elif r.intersects(circle): return "y"
    return "k"

def alpha(r,circles):
    for circle in circles:
        if r.intersects(circle.boundary): return 0.9
        elif r.intersects(circle): return 0.5
    return 0

start = time.time()
fig = plt.figure()

times = []
# for c in circles:
    # plt.gca().plot(*c.exterior.xy)
for r in tassel:
    t1 = time.time()
    plt.gca().fill(*zip(*np.array(list(zip(r.boundary.coords.xy[0][:-1], r.boundary.coords.xy[1][:-1])))),
                   'r', alpha= alpha(r,circles))
    t2 = time.time()
    times.append(t2-t1)

plt.axis('off')
end = time.time()
print('\nThe time for plotting is {:.2f} s.'.format(end - start))
#255.37 s

fig.savefig('circle_different_alphas.png', bbox_inches='tight',dpi=300)
