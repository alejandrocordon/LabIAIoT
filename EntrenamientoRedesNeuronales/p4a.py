from decimal import Decimal
import numpy as np
import NeuralNetworkLearning.debugInitializeWeights as diw
import NeuralNetworkLearning.computeNumericalGradient as cng
import NeuralNetworkLearning.checkNNGradients as cknng
import NeuralNetworkLearning.nnCostFunction as nncost


# Defining sigmoid function:
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoidGradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


def main():
    resultado = cknng.checkNNGradients(nncost)
    print(resultado)


main()
print('terminado')
