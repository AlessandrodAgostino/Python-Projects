#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:04:59 2019

@author: alessandro
"""
import pylab as plt

class Cell:
    
    def __init__(self, x, y):        
        self.x = x
        self.y = y
        self.state = 0
    
    def get_position(self):
        return (self.x, self.y)
        
    def get_state(self):
        return self.state
         
    def count_alive_neighbor(self, world):
        #It's suppose to be called only for the cell enough far from the boundaries
        count = 0
        for l in range(self.x-1,self.x+2):
            for m in range(self.y-1, self.y+2):
                count += world[l][m].get_state()
        count -= self.get_state()
        return count
    
    def forecast_state(self, world):
        num_alive_neigh = count_alive_neighbor(self, world)
        if self.state == 0:
            if num_alive_neigh < 2: 
                return 0
            elif num_alive_neigh > 3: return 0
            else: return 1
        elif num_alive_neigh == 3: return 1
        else: return 0   

class World:

    def __init__(self, dim):
        #Create a square world of dimension "dim" with DEAD cells
        self.dim = dim
        self.ground = [[] for i in range(dim+2)]
        for i in range(dim+2):
            for j in range(dim+2):
                self.ground[i][j] = Cell(i,j)
    
    def evolve(self):
        future_world = World(self.dim)
        for i in range(1,self.dim+2):
            for j in range(1,self.dim+2):
                future_world.ground[i][j] = self.ground[i][j].forecast_state(self)
        self.ground = future_world.ground
        
    
     
#%%
mondo = World(10)
        
        
        
        
        
    
    