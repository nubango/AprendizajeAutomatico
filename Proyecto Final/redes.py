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
num_etiquetas = 2

print(X.shape)
print(y.shape)

# Convertimos y en una matriz de 1 y 0
y_0 = np.where(y == 0, 1, 0)
y_1 = np.where(y == 1, 1, 0)
Y = np.vstack((y_0, y_1))
Y = Y.T
print(Y.shape)


def parte_1():
    print("\nParte 1 ----------------------------------\n")
    num_iter = 600

    theta1 = np.zeros((num_ocultas, num_entradas + 1))
    theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    Lambda = 0.0001

    percentage = ml.neural_network_training(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, Y, Lambda, num_iter)
    print("\nNeural Network (Lambda: {} | Iterations: {}) success rate: {}%".format(Lambda, num_iter, percentage))

def parte_2():
    print("\nParte 2 ----------------------------------\n")
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

def parte_3():
    print("\nParte 3 ----------------------------------\n")
    num_iter = 600

    theta1 = np.zeros((num_ocultas, num_entradas + 1))
    theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    lambdas = np.array([0.001, 0.01, 0.1])
    percentages = np.zeros(len(lambdas))

    for i in range(len(lambdas)):
        percentages[i] = ml.neural_network_training(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, Y, lambdas[i], num_iter)

    plt.plot(lambdas, percentages, color="purple")
    plt.xlabel("Lambda")
    plt.ylabel("% de acierto")
    plt.show()

    print("Success rates: {}".format(percentages))


# Hacemos un experimento base con lambda = 0.0001 y 600 iteraciones
parte_1()

# Hacemos un experimento con lambda = 1, 5, 10, 25, 50, 100
parte_2()

parte_3()