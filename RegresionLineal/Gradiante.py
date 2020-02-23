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
    
        
    print(costes)
    print(Thetas)
    
    
    plt.scatter(X[:, 1], Y, marker='+', color='red')
    plt.plot(X[:, 1], Thetas[0] + Thetas[1] * X[:, 1], linestyle='-', color='blue')
    plt.show()

    
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