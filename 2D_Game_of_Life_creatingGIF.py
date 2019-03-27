#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:22:25 2019

@author: alessandro
"""

#%%
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#First version of the evolution rule
def sing_evolution(value):
    if value in [3,13,14]: return 1.
    else: return 0.    

#Vectorize the function to use it on a matrix
vec_evolution = np.vectorize(sing_evolution)

def evolution(matrix):
    #Evaluate the argument for determining the evolution
    uniform3_filtered = ndimage.uniform_filter(matrix, size=3, mode = 'wrap')
    #Round the value to assign the evolution
    image_to_evolve = np.around(uniform3_filtered*9 + matrix*10)
    #Return the evolution
    return vec_evolution(image_to_evolve)

#---------------------------------------------------------------#
    
#Create a random matrix of float in [0,1)
ran_world = np.random.rand(50,50)
#Binarize the matrix
world = np.around(ran_world)

history = []
ims = []

fig = plt.figure()
for i in range(500):
    history.append(world)
    world = evolution(world)
    im = plt.imshow(world, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
ani.save("history.mp4")
    
