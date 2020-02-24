import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt


def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = sigmoid((H - Y) ** 2)
    return Aux.sum() / (2 * len(X))


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def pinta_frontera_recta(X, Y, theta):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera.pdf")
    plt.close()


def visualizacionDatos(X, Y):
    pos = np.where(Y == 1)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    pos = np.where(Y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='y')
    plt.show()
    plt.clf()


def descenso_gradiente(X, Y, alpha, array_costes):
    stop = 0.1
    Theta = np.array([0., 0.])
    i = 0
    for i in range(1500):
        Theta = gradiente(X, Y, Theta, alpha)
        costes = coste(X, Y, Theta)
        array_costes.append(costes)
        print(Theta, costes)

    return Theta, costes


def main():
    datos = carga_csv('ex2data1.csv')
    X = datos[:, :-1]
    Y = datos[:, -1]
    visualizacionDatos(X, Y)




main()
print('terminado')
