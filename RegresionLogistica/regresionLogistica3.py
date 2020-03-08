import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
# While using (return 1 / (1 + np.exp(-z))), per the sigmoid function, I was getting an overflow warning.
# As a solution warning can be ignored, or the dtype can be changed to not cause the error/warning.
# I used expit method from scipy to eliminate this issue.
from scipy.special import expit
# Importing minimize from scipy:
from scipy.optimize import minimize
# Importing PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures


def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


'''def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s'''


# Defining sigmoid function:
def sigmoid(z):
    # return 1 / (1 + np.exp(-z))
    return expit(z)


def pinta_frontera_recta(X, Y, theta):
    # plt.figure()
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
    # plt.savefig("frontera.png")
    plt.show()
    plt.clf()
    # plt.close()


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
    print('coste:', cost)
    return cost


# Creating plotData method to display the figure where the axes are the two exam scores.
def plotData(x, y, xlabel, ylabel, labelPos, labelNeg):
    # Separating positive and negative scores (in this case 1 and 0 values):
    pos = y == 1
    neg = y == 0

    # Scatter plotting the data, filtering them according the pos/neg values:
    plt.scatter(x[pos, 0], x[pos, 1], s=30, c='darkblue', marker='+', label=labelPos)
    plt.scatter(x[neg, 0], x[neg, 1], s=30, c='yellow', marker='o', edgecolors='b', label=labelNeg)

    # Labels and limits:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x[:, 0].min(), x[:, 0].max())
    plt.ylim(x[:, 1].min(), x[:, 1].max())

    # Legend:
    pst = plt.legend(loc='upper right', frameon=True)
    pst.get_frame().set_edgecolor('k')


# Defining costFunction method:
def costFunction(theta, X, y):
    # Number of training examples
    m = len(y)

    # eps = 1e-15  was taken from the solution by jellis18
    # https://github.com/jellis18/ML-Course-Solutions/blob/master/ex2/ex2.ipynb
    # It is tolerance for sigmoid function, fixes loss of precision error.
    # Eliminates errors while using BFGS minimization in calculations using scipy.
    eps = 1e-15

    hThetaX = sigmoid(np.dot(X, theta))

    J = - (np.dot(y, np.log(hThetaX)) + np.dot((1 - y), np.log(1 - hThetaX + eps))) / m

    return J


# Defining gradientFunc:
def gradientFunc(theta, X, y):
    # Number of training examples
    m = len(y)

    hThetaX = sigmoid(np.dot(X, theta))

    gradient = np.dot(X.T, (hThetaX - y)) / m

    return gradient


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


def predict(theta, X):
    hThetaX = sigmoid(np.dot(X, theta))

    arr = []
    for h in hThetaX:
        if (h > 0.5):
            arr.append(1)
        else:
            arr.append(0)

    return np.array(arr)


# Defining regularized costFunction method:
def costFunctionR(theta, X, y, lam):
    # Number of training examples
    m = len(y)

    eps = 1e-15

    hThetaX = sigmoid(np.dot(X, theta))

    J = - (np.dot(y, np.log(hThetaX)) + np.dot((1 - y), np.log(1 - hThetaX + eps)) -
           1 / 2 * lam * np.sum(np.square(theta[1:]))) / m

    return J


# Defining regularized gradientFunc:
def gradientFuncR(theta, X, y, lam):
    # Number of training examples
    m = len(y)

    hThetaX = sigmoid(np.dot(X, theta))

    # We're not regularizing the parameter θ0, replacing it with 0
    thetaNoZeroReg = np.insert(theta[1:], 0, 0)

    gradient = (np.dot(X.T, (hThetaX - y)) + lam * thetaNoZeroReg) / m

    return gradient


def plotDecisionBoundary(X, y, title, poly, result2):
    # Plot the data
    plotData(X[:, 1:3], y, 'Microchip Test 1', 'Microchip Test 2', 'Accepted', 'Rejected')

    # Defining the data to use in the meshgrid calculation. Outputting xx and yy ndarrays
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = sigmoid(poly.fit_transform(np.c_[xx.ravel(), yy.ravel()]).dot(result2['x']))
    Z = Z.reshape(xx.shape)

    # Plotting the contour plot
    plt.contour(xx, yy, Z, [0.5], linewidths=1, colors='g')
    plt.title(title)


def main():
    datos = carga_csv('ex2data2.csv')
    X, y = datos[:, :2], datos[:, 2]

    # Viewing the imported values (first 5 rows)
    X[:5], y[:5]

    plotData(X, y, 'Microchip Test 1', 'Microchip Test 2', 'Accepted', 'Rejected')

    # Creating the model
    poly = PolynomialFeatures(6)

    # Transforming the data into the sixth power polynomial
    X2 = poly.fit_transform(X)
    X2.shape

    # We add theta and initialize the parameters to 0's.
    initial_theta = np.zeros(X2.shape[1])
    initial_theta

    J = costFunctionR(initial_theta, X2, y, 1)
    gradient = gradientFuncR(initial_theta, X2, y, 1)

    # We should see that the cost is about 0.693 per the exercise:
    print("Cost: %0.3f" % J)
    print("Gradient: {0}".format(gradient))

    fig = plt.figure(figsize=(3, 1))

    # Creating 3 subplots using 3 different λ values
    for i, lam in enumerate([0, 1, 100]):
        result2 = minimize(costFunctionR, initial_theta, args=(X2, y, lam), method='BFGS', jac=gradientFuncR,
                           options={'maxiter': 400, 'disp': False})

        if (lam == 0):
            title = 'No regularization (Overfitting) (λ = 0)'
        elif (lam == 100):
            title = 'Too much regularization (Underfitting) (λ = 100)'
        else:
            title = 'Training data with decision boundary (λ = 1)'

        plt.subplot(3, 1, i + 1)

        # Plotting the decision boundary plot
        plotDecisionBoundary(X2, y, title, poly, result2);


main()
print('terminado')
