import numpy as np
import time
import matplotlib.pyplot as plt
from shapely.ops import polygonize,unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
from SALib.sample import saltelli

def color(r,circles):
    """Return the color of the region `r` depending on whether it intersects the
    boundary, the interior or not intersects any circle in `circles`"""
    for circle in circles:
        if r.intersects(circle.boundary): return "r"
        elif r.intersects(circle): return "y"
    return "k"

def alpha(r,circles):
    """Return the shade of the region `r` depending on whether it intersects the
    boundary, the interior or not intersects any circle in `circles`"""
    for circle in circles:
        if r.intersects(circle.boundary): return 0.9
        elif r.intersects(circle): return 0.5
    return 0

#%% PREPARING ALL THE  VORONOI ~ 20 s
#Low discrepancy sampling of the plane:
problem = {'num_vars': 2,
           'names': ['x', 'y'],
           'bounds': [[0, 100],[0, 100]]}

low_points = saltelli.sample(problem, 2000)
#0.23s

vor = Voronoi(low_points)
#0.85s

#CUTTING the global Voronoi tassellation to avoid divergences
#TO DO: it works but I don't know what it really does
lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line ]
convex_hull = MultiPoint([Point(i) for i in low_points]).convex_hull.buffer(2)
tassel = MultiPolygon([poly.intersection(convex_hull) for poly in polygonize(lines)])
tassel = MultiPolygon([p for p in tassel] + [p for p in convex_hull.difference(unary_union(tassel))])
#14.52 s


#%% CREATING THE CIRCLES AND PLOTTING THE VORONOI TASSELLATION WITH INTERECTIONS ~ 300 s
#Dictionary with the identity of all the circles in the image
circles_id = [{'r' : 10, 'c' : (40,30)},
              {'r' : 5, 'c' : (70,90)},
              {'r' : 15, 'c' : (20,60)},
              {'r' : 13, 'c' : (70,30)},
              {'r' : 20, 'c' : (0,0)}]

#List of the corrispondent polygons
circles = []
circles_b = []

thetas = np.linspace(0,2*np.pi,15)
for c_id in circles_id:
    x = np.cos(thetas)*c_id['r'] + c_id['c'][0]
    y = np.sin(thetas)*c_id['r'] + c_id['c'][1]
    circle_point = np.stack((x,y)).T
    xe = np.cos(thetas)*(c_id['r']+0.7)+ c_id['c'][0]
    ye = np.sin(thetas)*(c_id['r']+0.7)+ c_id['c'][1]
    x = np.append(x,xe, axis = 0)
    y = np.append(y,ye, axis = 0)
    bound_point = np.stack((x,y)).T
    circles.append((Polygon(circle_point)))
    circles_b.append((Polygon(bound_point)))


# for c in circles:
    # plt.gca().plot(*c.exterior.xy)
#plt.scatter(low_points[:,0], low_points[:,1],s=0.005, c="m", marker = "8")
# for r in tassel[0:1000]:
#     bound =  r.boundary.coords.xy
#     for circle, circle_b in zip(circles, circles_b):
#         if r.intersects(circle_b):
#             alpha=0.7
#             plt.gca().fill(r.boundary.coords.xy[0], r.boundary.coords.xy[1],'r', alpha= alpha, edgecolor='k', lw = 0.05)
#             nucleus = (np.sum(bound[0])/len(bound[0]),np.sum(bound[1])/len(bound[1]))
#             plt.gca().scatter(*nucleus,s=1, c="k", marker = ".", linewidth=0)
#
#         elif r.intersects(circle):
#             alpha=0.4
#             plt.gca().fill(r.boundary.coords.xy[0], r.boundary.coords.xy[1],'r', alpha= alpha, edgecolor='k', lw = 0.05)
#             nucleus = (np.sum(bound[0])/len(bound[0]),np.sum(bound[1])/len(bound[1]))
#             plt.gca().scatter(*nucleus,s=1, c="k", marker = ".", linewidth=0)
#
#         else: plt.gca().fill(r.boundary.coords.xy[0], r.boundary.coords.xy[1],'g', alpha= 0.5, edgecolor='k', lw = 0.05)
#

start = time.time()
fig = plt.figure()

for r in tassel:
    bound = r.boundary.coords.xy
    if any(r.intersects(circle_b) for circle_b in circles_b):
        plt.gca().fill(bound[0], bound[1],
                      'r', alpha= 0.7, edgecolor='k', lw = 0.05)
    elif any(r.intersects(circle) for circle in circles):
        plt.gca().fill(bound[0], bound[1],
                      'r', alpha= 0.4, edgecolor='k', lw = 0.05)

    else: plt.gca().fill(bound[0], bound[1],
                        'w', alpha= 0.7, edgecolor='k', lw = 0.05)

    nucleus = (np.sum(bound[0])/len(bound[0]),np.sum(bound[1])/len(bound[1]))
    plt.gca().scatter(*nucleus,s=2, c="k", marker = ".", linewidth = 0)

plt.axis('off')
end = time.time()
print('\nThe time for plotting is {:.2f} s.'.format(end - start))
#255.37 s

fig.savefig('circle_different_alphas_nuclei_2.png', bbox_inches='tight',dpi=1000)
