import numpy as np
import matplotlib.pyplot as plt

# calcula la integral de fun entre a y b por el metodo montecarlo
# generando para ello num_puntos aleatoriamente

# comparar tiempos de ejecucion entre ambas versiones

#calcular f(x) en muchos puntos para calcular el rectangulo que tenga dentro la funcion
#calcular que parte queda debajo
#calcular muchos puntos x,y aleatorios y mirar si estan por debajo o por encima de la curva mirando si y es >< que f(x)
#repetir 1M o 2M de veces y sacar % del area que corresponde a la integral

# version iterativa que realiza num_puntos iteraciones para calcular el resultado
def integra_mc(fun, a, b, num_puntos = 1000):
    div = (b-a)/1000
    M = 0
    for i in range(1000):
        if fun(a+div*i) > M:
            M = fun(a+div*i)   
    print(M)

def f(x):
    return x**x+1

def integra_mc_vec(fun, a, b, num_puntos = 1000):
    """Calcula la integral de f entre a y b por el metodo de
    Monte Carlo sin usar bucles"""
    eje_x = np.linspace(a, b, num_puntos)
    eje_y = fun(eje_x)
    max_f = max(eje_y)
    min_f = 0
    random_x = np.random.uniform(a, b, num_puntos)
    random_y = np.random.uniform(min_f, max_f, num_puntos)
    f_random_x = fun(random_x)

    plt.figure()
    plt.plot(random_x, random_y, 'x', c = 'red')
    plt.plot(eje_x, eje_y, '-')
    plt.savefig('mc.pdf')
    plt.close
    debajo = sum(random_y < f_random_x)
    area_total = abs(b-a) * abs(max_f - min_f)
    return area_total * (debajo / num_puntos)

integra_mc(f,0, 2)