import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def dot_product(x1, x2):
    """Calcula el producto escalar con un bucle y devuelve el tiempo en milisegundos"""
    tic = time.process_time()
    dot = 0
    for i in range(len(x1)):
        dot += x1[i] * x2[i]
    toc = time.process_time()
    return 1000 * (toc - tic)


def fast_dot_product(x1, x2):
    """Calcula el producto escalar vectorizado y devuelve el tiempo en milisegundos"""
    tic = time.process_time()
    dot = np.dot(x1, x2)
    toc = time.process_time()
    return 1000 * (toc - tic)


def compara_tiempos():
    sizes = np.linspace(100, 10000000, 20)
    times_dot = []
    times_fast = []
    for size in sizes:
        x1 = np.random.uniform(1, 100, int(size))
        x2 = np.random.uniform(1, 100, int(size))
        times_dot += [dot_product(x1, x2)]
        times_fast += [fast_dot_product(x1, x2)]

    plt.figure()
    plt.scatter(sizes, times_dot, c='red', label='bucle')
    plt.scatter(sizes, times_fast, c='blue', label='vector')
    plt.legend()
    plt.savefig('time.png')


def integra_mc(fun, a, b, num_puntos=10000):
    num_puntos = np.random.randint(a, b, (1, 1))
    integrate.quad(fun, a, b)


def fx(x):
    return x ^ 2;


# compara_tiempos()
integra_mc(fx(5), -2, 2, 5000)
print('terminado')
