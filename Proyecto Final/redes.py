import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import ML_utilities as ml

df = pd.read_csv("abalone_original.csv")

new_df = df.copy()

new_df['newRings_1'] = np.where(df['rings'] <= 8,1,0)
new_df['newRings_2'] = np.where(((df['rings'] > 8) & (df['rings'] <= 10)), 2,0)
new_df['newRings_3'] = np.where(df['rings'] > 10,3,0)

new_df['newRings'] = new_df['newRings_1'] + new_df['newRings_2'] + new_df['newRings_3']

X = new_df.drop(['sex','newRings', 'rings','newRings_1','newRings_2','newRings_3'], axis = 1)
y = new_df['newRings']

X.info()

m = X.shape[0]
num_entradas = 7
num_ocultas = 4
num_etiquetas = 3

print(X.shape)
print(y.shape)

# Convertimos y en una matriz de 1 y 0
y_0 = np.where(y == 1, 0, 0)
y_1 = np.where(y == 0, 1, 0)
y_2 = np.where(y == 0, 0, 1)
Y = np.vstack((y_0, y_1, y_2))
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

# Hacemos otro experimento con lambda = 0.001, 0.01, 0.1
parte_3()