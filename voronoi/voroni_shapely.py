import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import polygonize,unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from scipy.spatial import Voronoi

#Creating and cutting the Voronoi tassellation
points = 100 * np.random.random_sample((50,2))
vor = Voronoi(points)
lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line ]
convex_hull = MultiPoint([Point(i) for i in points]).convex_hull.buffer(2)
result = MultiPolygon([poly.intersection(convex_hull) for poly in polygonize(lines)])
result = MultiPolygon([p for p in result] + [p for p in convex_hull.difference(unary_union(result))])
result
#Creating the Circle
thetas = np.linspace(0,2*np.pi,15)
radius = 10
center = np.array([40,30])
x = np.cos(thetas)*radius + center[0]
y = np.sin(thetas)*radius + center[1]
circle_point = np.stack((x,y)).T
Circle = Polygon(circle_point)

fig = plt.figure()
plt.gca().plot(*Circle.exterior.xy)
for r in result:
    plt.gca().fill(*zip(*np.array(list(zip(r.boundary.coords.xy[0][:-1], r.boundary.coords.xy[1][:-1])))),
            "{}".format("w" if r.intersects(Circle.boundary) else "k"),
            alpha=0.4)
#plt.axis('off')

fig.savefig('circle_intersect_Voronoi_tight.png', bbox_inches='tight',dpi=300)
