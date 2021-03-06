import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from scipy.io import loadmat

def loadMat(fileName):
    return loadmat(fileName)

# Selecciona aleatoriamente 10 ejemplos y los pinta
def graphics(X):
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def cost(Theta, X, Y, l):
    return (-(((np.log(sigmoid(X.dot(Theta)))).T.dot(Y) 
+ (np.log(1 - sigmoid(X.dot(Theta)))).T.dot(1 - Y)) / X.shape[0])
    + (1 / (2 * X.shape[0])) * (Theta[1:,]**2).sum())


def gradient(Theta, X, Y, l):
    AuxO = np.hstack([np.zeros([1]), Theta[1:,]])
    return (((X.T.dot(sigmoid(X.dot(Theta)) - np.ravel(Y))) / X.shape[0]) 
    + (l / X.shape[0]) * AuxO)


def oneVsAll(X, Y, numEtiquetas, reg):
    clase = 10
    Theta = np.zeros([numEtiquetas, X.shape[1]])

    for i in range(numEtiquetas):
        claseVector = (Y == clase)

        result = opt.fmin_tnc(func = cost, x0 = Theta[i], fprime = gradient, args = (X, claseVector, reg))
        Theta[i] = result[0]

        if clase == 10:
            clase = 1
        else:
            clase += 1
    return Theta

# Determina el porcentaje de aciertos de la regresion logistica multicapa comparando los resultados estimados con los resultados reales
def logisticSuccessPercentage(X, Y, Theta):
    numAciertos = 0

    for i in range(X.shape[0]):
        results = sigmoid(X[i].dot(Theta.T))
        maxResult = np.argmax(results)
        if maxResult == 0:
            maxResult = 10
        if maxResult == Y[i]:
            numAciertos += 1
    return (numAciertos/(X.shape[0])) * 100

def neuronalSuccessPercentage(results, Y):
    numAciertos = 0
    for i in range(results.shape[0]):
        result = np.argmax(results[i]) + 1
        if result == Y[i]: numAciertos += 1
    return (numAciertos / (results.shape[0])) * 100

def propagacion(X1, Theta1, Theta2):
    X2 = sigmoid(X1.dot(Theta1.T))
    X2 = np.hstack([np.ones([X2.shape[0], 1]), X2])
    return sigmoid(X2.dot(Theta2.T))

def main():
    valores = loadMat("ex3data1.mat")

    X = valores['X']
    Y = valores['y']

    m = X.shape[0] 
    n = X.shape[1] 
    numEtiquetas = 10

    X = np.hstack([np.ones([m, 1]), X])

    # Cuanto mas se aproxime a 0, mas se ajustara el polinomio (menor regularizacion)
    l = 0.1
    Theta = oneVsAll(X, Y, numEtiquetas, l)

    success = logisticSuccessPercentage(X, Y, Theta)
    print("Precisión de la regresión logística: " + str(success) + " %")

    # Redes neuronales
    weights = loadMat('ex3weights.mat')
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']

    success = neuronalSuccessPercentage(propagacion(X, Theta1, Theta2), Y)
    print("Precisión de la red neuronal: " + str(success) + " %")

    graphics(X[:, 1:])

main()