import time
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def coste(Theta, X, Y, lmb):
    H = sigmoid(np.matmul(X, Theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) +
                               np.dot((1 - Y), np.log(1 - H))) + lmb/(2*len(X)) * (Theta[1:].T@Theta[1:])
    return cost

def gradiente(Theta, XX, Y, lmb):
    H = sigmoid(np.matmul(XX, Theta))
    grad1 = (1/len(Y))*(XX.T @ (H-Y))[0]
    grad2 = (1/len(Y))*(XX.T @ (H-Y))[1:] + (lmb/len(Y)) * Theta[1:]
    grad = np.hstack([grad1, grad2])
    return grad

def oneVsAll(X, y, lmb):
    num_etiquetas = 10
    n = np.shape(X)[1]
    allthetas = np.zeros((num_etiquetas, n))

    for j in range(num_etiquetas):
        result = opt.fmin_tnc(func=coste,
                              x0=allthetas[j],
                              fprime=gradiente,
                              args=(X,(y == (j+1)).flatten(), lmb))
        allthetas[j] = result[0]
    return allthetas

data = loadmat('Practicas/Practica3/datos/ex3data1.mat')
y = data['y']
X = data['X']

'''sample = np.random.choice(X.shape[0], 10)
plt.imshow(X[sample,:].reshape(-1,20).T)
plt.axis('off')
plt.show()'''

m = len(y)
ones = np.ones((m, 1))
X = np.hstack((ones, X))  # add the intercept
#(m, n) = X.shape
lmb = 0.1

theta = oneVsAll(X, y, lmb)
a = X @ theta.T
pred = np.argmax(X @ theta.T, axis=1)+1
acierto = np.mean(pred == y.flatten()) * 100
print((X @ theta.T)[1])
print(acierto)
