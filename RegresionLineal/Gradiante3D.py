# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:14:05 2020

@author: xustd
"""

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
 devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def main():
    datos = carga_csv('ex1data1.csv')  
    X = datos[:, :-1]
    #print(X)
    np.shape(X)         # (97, 1)
    Y = datos[:, -1]
    #print(Y)
    np.shape(Y)         # (97,)
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    # añadimos una columna de 1's a la X
    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01
    Thetas, costes = descenso_gradiente(X, Y, alpha)
    
    
    make_data([-10,10], [-1,4], X, Y,Thetas)
    
    
    
    
    #plt.plot(Thetas[0],Thetas[1],"x", c="red")
    #plt.clf()
    
    #plt.figure()
    #plt.scatter(X, Y, alpha)
    #plt.plot([5, 25], [Thetas[0] + Thetas[1]*5, Thetas[0] + Thetas[1]*25], c='red')
    #plt.show()
    #plt.contour(Thetas[0], Thetas[1], costes)
    #plt.contour(Thetas[0], Thetas[1], costes, np.logspace(-2, 3, 20), colors='blue')
    
def descenso_gradiente(X, Y, alpha):
    theta = np.zeros(2)
    for i in range(1500):
        newTheta = gradiente1(X, Y, theta, alpha)
        newCoste = coste(X, Y, newTheta)
    
    
    return [newTheta,newCoste]     
        
def coste(X, Y, Theta):
    H = np.dot(X, Theta) 
    #print(H)
    Aux = (H-Y)** 2
    return Aux.sum()/(2*len(X))

def make_data(t0_range, t1_range, X, Y, Thetas):
    """Genera las matrices X,Y,Z para generar un plot en 3D
    """
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    # Theta0 y Theta1 tienen las misma dimensiones, de forma que
    # cogiendo un elemento de cada uno se generan las coordenadas x,y
    # de todos los puntos de la rejilla
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])
        
    print(Coste)
    print(Theta0)
    
    
    X, Y = np.meshgrid(X, Y)
    plt.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20), colors='blue')
    plt.plot(Thetas[0],Thetas[1],"x", c="red")
    #plt.plot(Thetas[0],Thetas[1],"x", c="red")
    #plt.contour(Theta0, Theta1, Coste)
    #plt.scatter(Theta0[0],Theta1[0],marker='+', color='red')
    #plt.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20), colors='blue')
    #plt.plot(Thetas[0],Thetas[1])
    #plt.show()
    
    return [Theta0, Theta1, Coste]

  
    
def gradiente(X, Y, Theta, alpha):
 NuevaTheta = Theta
 m = np.shape(X)[0]
 n = np.shape(X)[1]
 H = np.dot(X, Theta)
 Aux = (H-Y)
 for i in range(n):
     Aux_i = Aux*X[:, i]
     NuevaTheta -= (alpha / m) * Aux_i.sum()
 return NuevaTheta


def gradiente1(X, Y, Theta, alpha):
 NuevaTheta = Theta
 m = np.shape(X)[0]
 n = np.shape(X)[1]
 H = np.dot(X, Theta)
 Aux = (H - Y)
 for i in range(n):
     Aux_i = Aux * X[:, i]
     NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
 return NuevaTheta
#def descenso_gradiente(X, Y, alpha):

main() 