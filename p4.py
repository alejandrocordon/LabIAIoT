import pandas as pd
import numpy as np
from decimal import Decimal


def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    # Set W to zero matrix
    W = np.zeros((fan_out, fan_in + 1))

    # Initialize W using "sin". This ensures that W is always of the same
    # values and will be useful in debugging.
    W = np.array([np.sin(w) for w in
                  range(np.size(W))]).reshape((np.size(W, 0), np.size(W, 1)))

    return W


def computeNumericalGradient(J, theta):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad


def checkNNGradients(costNN, reg_param):
    """
    Creates a small neural network to check the back propogation gradients.
    Outputs the analytical gradients produced by the back prop code and the
    numerical gradients computed using the computeNumericalGradient function.
    These should result in very similar values.
    """
    # Set up small NN
    num_entradas = 3
    num_ocultas = 5
    num_etiquetas = 3
    m = 5

    # Generate some random test data
    Theta1 = debugInitializeWeights(num_ocultas, num_entradas)
    Theta2 = debugInitializeWeights(num_etiquetas, num_ocultas)

    # Reusing debugInitializeWeights to get random X
    X = debugInitializeWeights(num_entradas - 1, m)

    # Set each element of y to be in [0,num_etiquetas]
    y = [(i % num_etiquetas) for i in range(m)]

    ys = np.zeros((m, num_etiquetas))
    for i in range(m):
        ys[i, y[i]] = 1

    # Unroll parameters
    params_rn = np.append(Theta1, Theta2).reshape(-1)

    # Compute Cost
    cost, grad = costNN(params_rn,
                        num_entradas,
                        num_ocultas,
                        num_etiquetas,
                        X, ys, reg_param)

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        return costNN(p, num_entradas, num_ocultas, num_etiquetas,
                      X, ys, reg_param)[0]

    numgrad = computeNumericalGradient(reduced_cost_func, params_rn)

    # Check two gradients
    #np.testing.assert_almost_equal(grad, numgrad)
    diff = Decimal(np.linalg.norm(numgrad-grad))/Decimal(np.linalg.norm(numgrad+grad))
    return (diff)

# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

def costNN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    Theta1 = params_rn[:((num_entradas+1) * num_ocultas)
                       ].reshape(num_ocultas, num_entradas+1)
    Theta2 = params_rn[((num_entradas + 1) * num_ocultas):].reshape(num_etiquetas, num_ocultas+1)

    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m, 1)), X))
    #y10 = np.zeros((m, num_etiquetas))

    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))  # hidden layer
    a2 = sigmoid(a1 @ Theta2.T)  # output layer

    y10 = y

    '''for i in range(1, num_etiquetas+1):
        y10[:, i-1][:, np.newaxis] = np.where(y == i, 1, 0)'''
    for j in range(num_etiquetas):
        J = J + sum(-y10[:, j] * np.log(a2[:, j]) -
                    (1-y10[:, j])*np.log(1-a2[:, j]))

    cost = 1/m * J
    reg_J = cost + reg / \
        (2*m) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))

    # Implement the backpropagation algorithm to compute the gradients

    grad1 = np.zeros((Theta1.shape))
    grad2 = np.zeros((Theta2.shape))

    for i in range(m):
        xi = X[i, :]  # 1 X 401
        a1i = a1[i, :]  # 1 X 26
        a2i = a2[i, :]  # 1 X 10
        d2 = a2i - y10[i, :]
        d1 = Theta2.T @ d2.T * sigmoidGradient(np.hstack((1, xi @ Theta1.T)))
        grad1 = grad1 + d1[1:][:, np.newaxis] @ xi[:, np.newaxis].T
        grad2 = grad2 + d2.T[:, np.newaxis] @ a1i[:, np.newaxis].T

    grad1 = 1/m * grad1
    grad2 = 1/m*grad2

    grad1_reg = grad1 + \
        (reg/m) * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    grad2_reg = grad2 + \
        (reg/m) * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    grad = np.hstack((grad1_reg.ravel(order='F'), grad2_reg.ravel(order='F')))

    return reg_J, grad

a = checkNNGradients(costNN, 0)

print(a)
