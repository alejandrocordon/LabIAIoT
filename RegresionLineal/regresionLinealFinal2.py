import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

x_max = 25
y_max = 25

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
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

def pintarPuntos(X, Y, Theta):
    plt.axis([0, x_max, 0, y_max])
    #plt.plot(X, Y, "r")
    plt.title(' Iteraciones')
    plt.scatter(X, Y, marker='+', color='red')
    #plt.scatter(X, Y)
    plt.plot(X, Theta[0] + Theta[1] * X, linestyle='-', color='blue')
    plt.savefig('mc.png')
    plt.show()
    plt.clf()


def pintarCostes(array_costes):
    i = 0
    for x in array_costes:
        i = i + 1
        plt.scatter(i, x, marker='+', color='red')
        #plt.plot(i,x)
    plt.savefig('costes.png')
    plt.show()
    plt.clf()


def pintarCurvas(Theta, Coste):
    plt.contour(Theta[0], Theta[1], Coste, np.logspace(-2, 3, 20), colors='blue')
    plt.savefig('curvas.png')
    plt.show()
    plt.clf()

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

    array_costes = []

    # añadimos una columna de 1's a la X
    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01
    Thetas, costes = descenso_gradiente(X, Y, alpha, array_costes)

    pintarPuntos(X[:, 1], Y, Thetas)
    pintarCostes(array_costes)

    Theta0, Theta1, Coste = make_data(x_max, y_max, X, Y)
    pintarCurvas(Theta0, Theta1, costes)

    print(Thetas)
    # a = make_data(m, n, X, Y) #X, Y, alpha)

main()
print('terminado')
