import numpy as np
import scipy
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat


def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)



def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def pinta_frontera_recta(X, Y, theta):
    #plt.figure()
    pos = np.where(Y == 1)
    plt.scatter(X[pos, 1], X[pos, 2], marker='+', c='k')
    pos = np.where(Y == 0)
    plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='y')

    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
                      xx1.ravel(),
                      xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    #plt.savefig("frontera.png")
    plt.show()
    plt.clf()
    #plt.close()



def visualizacionDatos(X, Y):
    pos = np.where(Y == 1)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    pos = np.where(Y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='y')
    plt.show()
    plt.clf()


def cost(theta, X, Y):
    # H = sigmoid(np.matmul(X, np.transpose(theta)))
    H = sigmoid(np.matmul(X, theta))
    # cost = (- 1 / (len(X))) * np.sum( Y * np.log(H) + (1 - Y) * np.log(1 - H))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    print('coste:',cost)
    return cost


def gradient(theta, XX, Y):
    H = sigmoid(np.matmul(XX, theta))
    grad = (1 / len(Y)) * np.matmul(XX.T, H - Y)
    print('gradiente:',grad)
    return grad



def plot_decisionboundary(X, Y, theta, poly):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(),
    xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.savefig("boundary.png")
    plt.close()


def main():
    datos = scipy.io.loadmat('ex3data1.mat')
    Y = datos['y']
    X = datos['X']
    #visualizacionDatos(X, Y)

    # Selecciona aleatoriamente 10 ejemplos y los pinta
    sample = np.random.choice(X.shape[0], 10 )
    plt.imshow(X[sample, :].reshape(-1, 20 ).T)
    plt.axis('off')
    plt.show()





main()
print('terminado')
