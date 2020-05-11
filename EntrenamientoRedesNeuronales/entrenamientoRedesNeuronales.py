import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
# While using (return 1 / (1 + np.exp(-z))), per the sigmoid function, I was getting an overflow warning.
# As a solution warning can be ignored, or the dtype can be changed to not cause the error/warning.
# I used expit method from scipy to eliminate this issue.
from scipy.io import loadmat
from scipy.special import expit
# Importing minimize from scipy:
from scipy.optimize import minimize
# Importing PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from EntrenamientoRedesNeuronales.displayData import displayData
from EntrenamientoRedesNeuronales.checkNNGradients import debugInitializeWeights
from EntrenamientoRedesNeuronales.checkNNGradients import debugInitializeWeights


def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def carga_mat(file_name):
    valores = loadmat(file_name)
    return valores


# Defining sigmoid function:
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def cost(theta, X, Y, lmb):
    # H = sigmoid(np.matmul(X, np.transpose(theta)))
    H = sigmoid(np.matmul(X, theta))
    # cost = (- 1 / (len(X))) * np.sum( Y * np.log(H) + (1 - Y) * np.log(1 - H))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H))) + lmb / (2 * len(X)) * (
            theta[1:].T @ theta[1:])
    print('coste:', cost)
    return cost


# Defining gradientFunc:
def gradientFunc(theta, X, y, lmb):
    H = sigmoid(np.matmul(X, theta))

    gradient1 = (1 / len(y)) * (X.T @ (H - y))[0]
    gradient2 = (1 / len(y)) * (X.T @ (H - y))[1:] + (lmb / len(y)) * theta[1:]
    gradient = np.hstack([gradient1, gradient2])

    return gradient


def oneVsAll(X, y, num_etiquetas, reg):
    """
    oneVsAll entrena varios clasificadores prregresión logística con término
    de regularización ’reg’ y devulveel resultado en una matriz , donde
    la fila i−ésima corresponde al clasificador de la etiquetai−ésima
    """
    n = np.shape(X)[1]
    thetas = np.zeros((num_etiquetas, n))

    for j in range(num_etiquetas):
        res = opt.fmin_tnc(func=cost, x0=thetas[j], fprime=gradientFunc, args=(X, (y == (j + 1)).flatten(), reg))
        thetas[j] = res[0]
    return thetas



def main():
    datos = carga_mat('/Users/alejandrocordonurena/LabIAIoT/RegresionLogisticaMulticlase/ex3data1.mat')
    y = datos['y']
    X = datos['X']

    # Selecciona aleatoriamente 10 ejemplos y los pinta
    sample = np.random.choice(X.shape[0], 10 )
    plt.imshow(X[sample, :].reshape(-1, 20 ).T)
    plt.axis('off')
    plt.show()

    m = len(y)
    ones = np.ones((m, 1))
    X = np.hstack((ones, X))  # add the intercept
    # (m, n) = X.shape
    reg = 0.1

    num_etiquetas = 10
    theta = oneVsAll(X, y, num_etiquetas, reg)
    a = X @ theta.T
    predict = np.argmax(X @ theta.T, axis=1) + 1
    acierto = np.mean(predict == y.flatten()) * 100
    print((X @ theta.T)[1])
    print(acierto)

    displayData(X)
    displayImage(im)


main()
print('terminado')
