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
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))


def make_data(t0_range, t1_range, X, Y):
    """Genera las matrices X,Y,Z para generar un plot en 3D"""
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
    return [Theta0, Theta1, Coste]


""" descenso gradiente 

 Es una funcion para minimizar funciones
 La idea es q hay q cambiar los valores de theta hasta que el coste sea minimo
 
 Usamos la derivada en un punto
 calculo la derivada del coste y eso dice donde debo ir
 Si todo va bien, el coste cada vez es menor.
 
 Es un método general de optimización
 
 
 """


def descenso_gradiente(X, Y, alpha):
    stop = 0.1
    Theta = np.array([0., 0.])
    i = 0
    # for i in range(1000):
    return gradiente(X, Y, Theta, alpha)
    coste_valor = coste(X, Y, NuevaTheta)
    print("NuevaTheta:" + NuevaTheta)
    # print("coste_valor:"+coste_valor)


"""
def gradiente(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    h = np.dot(X, Theta)
    aux = (h - Y)
    for i in range(1000):
        Aux_i = aux * X[:, i]
        NuevaTheta -= (alpha / m) * Aux_i.sum()
    return NuevaTheta
"""


def gradiente(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
    return NuevaTheta


"""
def pintarPuntos(X, Y):
    plt.scatter(X, Y, marker='+', color='red')
    plt.savefig('mc.png')
    # plt.show()
    plt.clf()
"""


def pintarPuntos(X, Y, Theta):
    plt.scatter(X, Y, marker='+', color='red')
    plt.plot(X, Theta[0] + Theta[1] * X, linestyle='-', color='blue')
    plt.savefig('mc.png')
    # plt.show()
    plt.clf()


"""Recta Theta0 + Theta1X """

"""
def pintarRecta(punto0, punto1):
    plt.scatter((0, 10), (h0, h10), marker='+', color='red')
    # plt.show()
    plt.clf()
"""


def main():
    # m = number of training examples
    # x = input
    # y = output
    # (x,y) = one training example
    # (xi, yi) = ith training example
    datos = carga_csv('ex1data1.csv')
    X = datos[:, :-1]
    np.shape(X)  # (97, 1)
    Y = datos[:, -1]
    np.shape(Y)  # (97,)
    m = np.shape(X)[0]
    n = np.shape(X)[1]

    # añadimos una columna de 1's a la X
    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01
    Thetas = descenso_gradiente(X, Y, alpha)

    pintarPuntos(X[:, 1], Y, Thetas)

    print(Thetas)
    # a = make_data(m, n, X, Y) #X, Y, alpha)


# ex1data1.csv Datos sobre los que aplicar regresión lineal con una variable.
# ex1data2.csv Datos sobre los que aplicar regresión lineal con varias variables.

main()
print('terminado')

# hacer un bucle que recorra los m ejemplos de entrenamiento

# theta 0 y theta 1 los inicializamos a 0 antes del bucle

# en vez de hacer un buble multiplicar la matriz theta por el vector X  - dot de un vector por otro


"""
m = Number of training examples
 x’s = “input” variable / features
 y’s = “output” variable / “target” variable
(x, y) = one training example
(x(i), y(i)) = ith training example
"""
