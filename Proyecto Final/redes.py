import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import ML_utilities as ml

df = pd.read_csv("abalone_original.csv")

df['newRings'] = np.where(df['rings'] > 10,1,0)

X = df.drop(['newRings','rings','sex'], axis = 1)
y = df['newRings']

X.info()

m = X.shape[0]
num_entradas = 7
num_ocultas = 7
num_etiquetas = 1

print(X.shape)
print(y.shape)

Y = y

print(Y.shape)


def prueba1():
    print("\nPrueba 1 ----------------------------------\n")
    num_iter = 600

    theta1 = np.zeros((num_ocultas, num_entradas + 1))
    theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    Lambda = 0.0001

    percentage = ml.neural_network_training(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, Y, Lambda, num_iter)
    print("\nNeural Network (Lambda: {} | Iterations: {}) success rate: {}%".format(Lambda, num_iter, percentage))

def prueba2():
    print("\nPrueba 2 ----------------------------------\n")
    num_iter = 600

    theta1 = np.zeros((num_ocultas, num_entradas + 1))
    theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    lambdas = np.array([1, 5, 10, 25, 50, 100])
    percentages = np.zeros(len(lambdas))

    for i in range(len(lambdas)):
        percentages[i] = ml.neural_network_training(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, Y, lambdas[i], num_iter)

    plt.plot(lambdas, percentages, color="red")
    plt.xlabel("Lambda")
    plt.ylabel("% de acierto")
    plt.show()

    print("Success rates: {}".format(percentages))

def prueba3():
    print("\nParte 3 ----------------------------------\n")
    num_iter = 600

    theta1 = np.zeros((num_ocultas, num_entradas + 1))
    theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    lambdas = np.array([0.001, 0.01, 0.1])
    percentages = np.zeros(len(lambdas))

    for i in range(len(lambdas)):
        percentages[i] = ml.neural_network_training(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, Y, lambdas[i], num_iter)

    plt.plot(lambdas, percentages, color="red")
    plt.xlabel("Lambda")
    plt.ylabel("% de acierto")
    plt.show()

    print("Success rates: {}".format(percentages))


# Hacemos un experimento base con lambda = 0.0001 y 600 iteraciones
prueba1()

# Hacemos un experimento con lambda = 1, 5, 10, 25, 50, 100
prueba2()

# Hacemos otro experimento con lambda = 0.001, 0.01, 0.1
prueba3()