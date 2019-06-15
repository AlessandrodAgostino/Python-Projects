import numpy as np
import matplotlib.pyplot as plt
thetas = np.linspace(0,2*np.pi,12)
radius = 0.2
center = np.array([0.4,0.4])
x = np.cos(thetas)*radius + center[0]
y = np.sin(thetas)*radius + center[1]
circle = np.stack((x,y))


f = plt.figure()
plt.plot(circle[0,:], circle[1,:])
plt.gca().set_ylim(-1,1)
plt.gca().set_xlim(-1,1)

#&&
