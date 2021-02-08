import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from scipy.io import loadmat

# Carga el fichero mat especificado y lo devuelve en una matriz data
def loadMat(fileName):
    return loadmat(fileName)

# Selecciona aleatoriamente 10 ejemplos y los pinta
def graphics(X):
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()

# g(X*Ot) = h(x)
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

# Devuelve un valor de coste
def cost(O, X, Y, l):
    return (-(((np.log(sigmoid(X.dot(0)))).T.dot(Y) 
    + (np.log(1 - sigmoid(X.dot(0)))).T.dot(1 - Y)) / X.shape[0])
    + (1 / (2 * X.shape[0])) * (O[1:,]**2).sum())

# La operacion que hace el gradiente por dentro. Devueve un vector de valores
def gradient(O, X, Y, l):
    AuxO = np.hstack([np.zeros([1]), O[1:,]])
    return (((X.T.dot(sigmoid(X.dot(O)) - np.ravel(Y))) / X.shape[0]) 
    + (l / X.shape[0]) * AuxO)

# Para cada ejemplo (fila de Xs), haya los pesos theta por cada posible tipo de numero que pueda ser
def oneVsAll(X, Y, numEtiquetas, reg):
    clase = 10
    O = np.zeros([numEtiquetas, X.shape[1]])

    for i in range(numEtiquetas):
        claseVector = (Y == clase)

        result = opt.fmin_tnc(func = cost, x0 = O[i], fprime = gradient, args = (X, claseVector, reg))
        O[i] = result[0]

        if clase == 10:
            clase = 1
        else:
            clase += 1
    return O

# Determina el porcentaje de aciertos de la regresion logistica multicapa comparando los resultados estimados con los resultados reales
def logisticSuccessPercentage(X, Y, O):
    numAciertos = 0

    for i in range(X.shape[0]):
        results = sigmoid(X[i].dot(O.T))
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

def propagacion(X1, O1, O2):
    X2 = sigmoid(X1.dot(O1.T))
    X2 = np.hstack([np.ones([X2.shape[0], 1]), X2])
    return sigmoid(X2.dot(O2.T))

def main():
    valores = loadMat("ex3data1.mat")

    X = valores['X']
    Y = valores['Y']

    m = X.shape[0] # numero de muestras de entrenamiento
    n = X.shape[1] # numero de variable x que influyen en el resultado y, mas la columna 1s 
    numEtiquetas = 10

    X = np.hs([np.one([m, 1]), X])

    # Cuanto mas se aproxime a 0, mas se ajustara el polinomio (menor regularizacion)
    l = 0.1
    O = oneVsAll(X, Y, numEtiquetas, l)

    # Redes neuronales
    weights = loadMat('ex3weights.mat')
    O1, O2 = weights['Theta1'], weights['Theta2']

    success = logisticSuccessPercentage(X, Y, O)
    print("Logistic regression success: " + str(success) + " %")

    graphics(X[:, 1:])

main()