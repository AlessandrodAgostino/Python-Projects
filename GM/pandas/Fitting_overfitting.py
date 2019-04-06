#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:16:32 2019

@author: alessandro
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pylab as plt 

x = np.linspace(-1,1,10)
y = np.sin(2*np.pi*x)
plt.plot(x,y,'o-')
plt.show
eps=0.1
y = np.sin(2*np.pi*x)+st.norm.rvs(size = len(x), loc=0,scale=eps)
#possiamo aggiungere il rumore

#%%
def datagen(x,eps):
    return np.sin(2*np.pi*x)+st.norm.rvs(size = len(x), loc=0,scale=eps)

def datagen2(x,f, args, eps):
    return f(x,args)+st.norm.rvs(size = len(x), loc=0,scale=eps)

x = np.linspace(-1,1,1000)
y = np.sin(2*np.pi*x)
plt.plot(x,y,'g',linewidth=2,label='true func')

X = np.linspace(-1,1,10)
eps=0.2
Y = datagen (X,eps)
plt.scatter(X,Y,s=140,label='mydata',facecolors='w',edgecolor='b')

m=8
p=np.polyfit(X,Y,m)
print(p)
#la funzione polinomio viene definita da poly1d -> 1 dimensionale.
#Prende in input i prametri, non l'incognita x.
yexp = np.poly1d(p)
plt.plot(x,yexp(x),'r--',label='poly '+str(m))
plt.legend()
plt.show


#for i,m in enumerate([0,1,3,9]):
#    plt.sublot(221+i)
#    p = np.polyfit(X, Y, m)
#    yexp = np.poly1d(p)
#    plt.scatter(X,Y, s=40, label='mydata')

#%%
X1 = np.linspace(0, 1, 10)
train = datagen(X1,eps)
X = np.linspace(0, 1, 10)
test = datagen(X,eps)

def RMS(y,yexp):
    s = np.sum((yexp-y)**2)
    n=len(y)
    return np.sqrt(s)/n #normalizzata

mmax = 25
testRMS = np.empty(mmax)
trainRMS = np.empty(mmax)
pp = np.zeros((mmax, mmax))
M = np.arange(mmax)
for m in M:
    p = np.polyfit(X1, train, m)
    for i in np.arange(m+1):
        pp[i,m] = p[i]

    fexp = np.poly1d(p)
    trainexp=fexp(X1)
    trainRMS[m] = RMS(train,trainexp)
    testexp=fexp(X)
    testRMS[m] = RMS(test, testexp)

plt.scatter(M, trainRMS, s=140, facecolor='w', edgecolor='b')
plt.scatter(M, testRMS, s=140, facecolor='w', edgecolor='r')
plt.plot(M, trainRMS, 'ob-', label='train')
plt.plot(M, testRMS, 'or-', label='test')
plt.ylabel('RMS', fontsize=20)
plt.xlabel('M', fontsize=20)
plt.legend(loc='best',fontsize=16)
plt.show()
#%%
myplotfit = pd.DataFrame(pp).apply(lambda x: np.round(x,3))
myplotfit.head()
#%%

for i,size in enumerate([10, 100]):
    plt.subplot(121+i)
    x = np.linspace(0,1,1000)
    plt.plot(x, np.sin(2*np.pi*x),'g', linewidth=2, label='true func')
    X1 = np.linspace(0,1,size)
    train = datagen(X1, eps)
    plt.scatter(X1, train, s=100)
    ####
    plt.plot(x, yexp(x), 'r--', label='poly '+str(m))
    plt.legend()
    
#%%
def sq_err(x,y,f,p=()):
    from scipy.optimize import minimize
    def cost(args):#i parametri vanno passati tutti insieme come un unico vettore
        return np.sum((y-f(x,args))**2)*0.5
    return minimize(cost,p) #restituisce un oggetto con vari attributi tra cui i valori dei parametri ottimizzati

def sq_err_ridge(x,y,f, lamb= 0,p=()):
    from scipy.optimize import minimize
    def cost(args):
        return np.sum((y-f(x,args))**2)*0.5 + lamb*np.sum(args**2)*0.5
    return minimize(cost,p) #restituisce un oggetto con vari attributi tra cui i valori dei parametri ottimizzati

def my_poly(x, args=()):
    f = np.poly1d(args)
    return f(x)

x = np.arange(10)
args = tuple(np.array([1,1,1,1,1,1]))
y = my_poly(x,args)
#print(x,y)
#plt.plot(x,y)
#plt.show()

res = sq_err(x,y,my_poly,args).x
print('standard:',res)

res = sq_err_ridge(x,y,my_poly,lamb=1,p=args).x
print('penalized:',res)

X = np.linspace(0,1,10)
eps= 0.3
X1 = np.linspace(0,1,size)



for i,size in enumerate([0,3,9,15]):
    plt.subplot(221+i)
    plt.plot(X, np.sin(2*np.pi*X),'g', linewidth=2, label='true func')
    train = datagen(X1, eps)
    plt.scatter(X1, train, s=100)
    ####
    plt.plot(x, yexp(x), 'r--', label='poly '+str(m))
    plt.legend()
#%%
X1 = np.linspace(0, 1, 10)
train = datagen(X1,eps)
X = np.linspace(0, 1, 10)
test = datagen(X,eps)    
    
lamb=1e-3
mmax = 25
testRMS = np.empty(mmax)
trainRMS = np.empty(mmax)
pp = np.zeros((mmax, mmax))
M = np.arange(mmax)

for m in M:
    p = sq_err_ridge(X1, train, my_poly, lamb=lamb, p=np.zeros(m+1)).x
    for i in np.arange(m+1):
        pp[i,m] = p[i]

    fexp = np.poly1d(p)
    trainexp=fexp(X1)
    trainRMS[m] = RMS(train,trainexp)
    testexp=fexp(X)
    testRMS[m] = RMS(test, testexp)

plt.scatter(M, trainRMS, s=140, facecolor='w', edgecolor='b')
plt.scatter(M, testRMS, s=140, facecolor='w', edgecolor='r')
plt.plot(M, trainRMS, 'ob-', label='train')
plt.plot(M, testRMS, 'or-', label='test')
plt.ylabel('RMS', fontsize=20)
plt.xlabel('M', fontsize=20)
plt.legend(loc='best',fontsize=16)
plt.show()
myplotfit = pd.DataFrame(pp).apply(lambda x: np.round(x,3))
myplotfit.head()

#%%
try_ =  np.arange(5,25)
lambda_ = np.arange(5,25)
    
#for j,l in enumerate(try_):
#    print(str(j)+" "+str(l))

for j,l in enumerate(lambda_):
    print(str(j)+" ciao "+str(l))








