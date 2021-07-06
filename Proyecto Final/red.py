import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def normaliza_datos(X):
    
    media = np.mean(X, axis=0)
    desviacion = np.std(X, axis = 0)
    Xnorm = (X-media) / desviacion

    return Xnorm

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def check_accuracy(y, out):
    max_i = np.argmax(out, axis = 1) +1
    #max_i = np.where(out >= 0.5)
    control = (y[:, 0] == max_i) 

    # Porcentaje sobre el total de ejemplos redondeado
    return 100 * np.size(np.where(control == True)) / y.shape[0]

def pesosAleatorios(L_in, L_out):
    epsilon = 0.12
    
    # Inicializamos la matriz con 0s
    pesos = np.zeros((L_out, 1 + L_in))
    
    # Valores aleatorios en ese intervalo
    pesos = np.random.rand(L_out, 1 + L_in) * (2 * epsilon) - epsilon
    return pesos

def forward_prop(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def back_propagation(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, Lambda):
    # Unroll thetas (neural network params)
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    A1, Z2, A2, Z3, A3 = forward_prop(X, theta1, theta2)

    # Cost function (without reg term)
    m = X.shape[0]
    cost_unreg_term = (-Y * np.log(A3) - (1 - Y) * np.log(1 - A3)).sum() / m

    # Cost function (with reg term)
    cost_reg_term = (Lambda / (2 * m)) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))
    cost = cost_unreg_term + cost_reg_term

    # Numerical gradient (without reg term)
    Theta1_grad = np.zeros(np.shape(theta1))
    Theta2_grad = np.zeros(np.shape(theta2))
    D3 = A3 - Y
    D2 = np.dot(D3, theta2)
    D2 = D2 * (np.hstack([np.ones([Z2.shape[0], 1]), sigmoidGradient(Z2)]))
    D2 = D2[:, 1:]
    Theta1_grad = Theta1_grad + np.dot(A1.T, D2).T
    Theta2_grad = Theta2_grad + np.dot(A2.T, D3).T

    # Numerical gradient (with reg term)
    Theta1_grad = Theta1_grad * (1 / m)
    Theta2_grad = Theta2_grad * (1 / m)
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (Lambda / m) * theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (Lambda / m) * theta2[:, 1:]
    grad = np.concatenate((np.ravel(Theta1_grad), np.ravel(Theta2_grad)))

    return (cost, grad)

def optimize(backprop, params_rn, input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda, num_iter):
    result = opt.minimize(fun=backprop, x0=params_rn,
    args=(input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda),
    method='TNC', jac=True, options={'maxiter': num_iter})
    return result.x

def neural_network_training(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda, num_iter):
    
    # Init Neural Network params
    theta1 = pesosAleatorios(theta1.shape[1] - 1, theta1.shape[0])
    theta2 = pesosAleatorios(theta2.shape[1] - 1, theta2.shape[0])

    # Train Neural Network
    params_rn = np.concatenate([theta1.reshape(-1), theta2.reshape(-1)])
    theta_opt = optimize(back_propagation, params_rn, input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda, num_iter)
    theta1_opt = np.reshape(theta_opt[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)))
    theta2_opt = np.reshape(theta_opt[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1 )))

    return check_accuracy(Y, forward_prop(X, theta1_opt, theta2_opt)[4])

############################################################

df = pd.read_csv("abalone_original.csv")

df['newRings'] = np.where(df['rings'] > 10,0,1)

y = df['newRings']
X = df.drop(['newRings','rings','sex'], axis = 1)
m = len(y)

Xnorm = normaliza_datos(X) 

X.info()

num_entradas = 7
num_ocultas = 4
num_etiquetas = 2

print(X.shape)
#y_one = y[:,np.newaxis]
#y_one = y[:,np.newaxis]
print(y.shape)

#Y = np.hstack([np.ones([m,1]),y)
y = (y-1)
y_one = np.zeros((m , num_etiquetas))

for i in range(m):
    y_one[i][y[i]] = 1

print(y_one.shape)


def prueba():
    print("\nPrueba ----------------------------------\n")
    num_iter = 600

    theta1 = np.zeros((num_ocultas, num_entradas + 1))
    theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    Lambda = 0.001

    percentage = neural_network_training(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, Xnorm, y_one, Lambda, num_iter)
    print("\nNeural Network (Lambda: {} | Iteraciones: {}) Porcentaje de acierto: {}%".format(Lambda, num_iter, percentage))

def prueba2():
    print("\nPrueba 2 ----------------------------------\n")
    num_iter = 600

    theta1 = np.zeros((num_ocultas, num_entradas + 1))
    theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    lambdas = np.array([5, 10, 25, 50, 100, 200])
    percentages = np.zeros(len(lambdas))

    for i in range(len(lambdas)):
        percentages[i] = neural_network_training(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, Xnorm, y_one, lambdas[i], num_iter)

    plt.plot(lambdas, percentages, color="red")
    plt.xlabel("Lambda")
    plt.ylabel("% de acierto")
    plt.show()

    print("Lambdas: ")
    for i in range(len(lambdas)):
        print(lambdas[i])
    print("Porcentajes de acierto: {}".format(percentages))

def prueba3():
    print("\nParte 3 ----------------------------------\n")
    num_iter = 600

    theta1 = np.zeros((num_ocultas, num_entradas + 1))
    theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    lambdas = np.array([0.01,0.05, 0.1, 0.5, 1])
    percentages = np.zeros(len(lambdas))

    for i in range(len(lambdas)):
        percentages[i] = neural_network_training(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, Xnorm, y_one, lambdas[i], num_iter)

    plt.plot(lambdas, percentages, color="red")
    plt.xlabel("Lambda")
    plt.ylabel("% de acierto")
    plt.show()

    print("Lambdas: ")  
    for i in range(len(lambdas)):
        print(lambdas[i])
    print("Porcentajes de acierto: {}".format(percentages))

prueba()
prueba2()
prueba3()