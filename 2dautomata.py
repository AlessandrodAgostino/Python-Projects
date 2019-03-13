import numpy as np

def generate_world(dim):
    world = np.zeros((dim,dim), dtype=int)
    #add constat DEAD boundaries
    np.pad(world, (2,2), 'constant', constant_values=(0, 0))
    return world

def count_alive_neigh(i,j, present):
    count = 0
    for l in range(i-1,i+2):
        for m in range(j-1, j+2):
            count += present[l][m]
    count -= present[i][j]
    return count

def evolution(present):
    future = [[]]
    future_line = []
    for i in range(2,len(present)-1):
        for j in range(2,len(present[i])-1):       
            num_alive_neigh = count_alive_neigh(i,j, present)                 
            life = survival(present[i,j], num_alive_neigh)
            future_line.append(life)
        future.append(future_line)
        
def survival(cell, num_alive_neigh):
    if cell :
        if num_alive_neigh < 2: 
            return 0
        elif num_alive_neigh > 3: return 0
        else: return 1
    elif num_alive_neigh == 3: return 1
    else: return 0

world = generate_world(10)  
evolution(world)

    
#%%
lista=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
print(lista)
print(lista[1:-1])

    
            
    
    
