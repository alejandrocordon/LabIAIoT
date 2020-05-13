from decimal import Decimal
import numpy as np
import NeuralNetworkLearning.debugInitializeWeights as diw
import NeuralNetworkLearning.computeNumericalGradient as cng
import NeuralNetworkLearning.nnCostFunction as nncost


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
    Theta1 = diw.debugInitializeWeights(num_ocultas, num_entradas)
    Theta2 = diw.debugInitializeWeights(num_etiquetas, num_ocultas)

    # Reusing debugInitializeWeights to get random X
    X = diw.debugInitializeWeights(num_entradas - 1, m)

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

    numgrad = cng.computeNumericalGradient(reduced_cost_func, params_rn)

    # Check two gradients
    # np.testing.assert_almost_equal(grad, numgrad)
    diff = Decimal(np.linalg.norm(numgrad - grad)) / Decimal(np.linalg.norm(numgrad + grad))
    return (diff)


def main():
    resultado = checkNNGradients(nncost.costNN, 0)
    print(resultado)


main()
print('terminado')
