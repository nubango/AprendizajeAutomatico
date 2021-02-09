import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file):
    valores = read_csv ( file ,  header=None) . to_numpy ()
    return valores.astype(float)

def normaliza_datos(X):
    
    media = np.zeros(X[1].size)
    desviacion = np.zeros(X[1].size)

    for x in range(X[1].size):
        xData = X[:,x]
        media[x] = np.median(xData)
        desviacion[x] = np.std(xData)

    xNormalized = (X - media) / desviacion
     
    return xNormalized, media, desviacion


def coste(X, Y, T):
    H = np.dot(X, T)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

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

def descenso_gradiente(X, Y, alpha, m, n):
    Thetas = np.zeros(X[1].size)

    for i in range(400):

        Thetas = Thetas - alpha * (1 / m) * ( X.T.dot(np.dot(X, Thetas) - Y) )

        coste(X, Y, Thetas)
        costes = []
        costes = np.append(costes, coste(X, Y, Thetas))

    return Thetas
    
def ecuacionNormal(X, Y):
     return np.linalg.inv(X.T @ X) @ X.T @ Y

def draw_cost(cost):
    """
    Draw the linear progression of the cost
    """
    plt.figure()
    X = np.linspace(0, 400, len(cost))
    plt.plot(X, cost)
    plt.show()

datos = carga_csv('ex1data2.csv')
X = datos[:, :-1]
np.shape(X)
Y = datos[:, -1]
np.shape(Y)
m = np.shape(X)[0]
n = np.shape(X)[1]
alpha = 0.01



xNorm, mu, sigma = normaliza_datos(X)

# añadimos una columna de 1's a la X
xNorm = np.hstack([np.ones([m, 1]), xNorm])

rectaCosteReducido = descenso_gradiente(xNorm, Y, alpha, m , n)

# añadimos una columna de 1's a la X
X = np.hstack([np.ones([m, 1]), X])

rectaEcuacionNormal = ecuacionNormal(X, Y)

print(rectaCosteReducido)
print(rectaEcuacionNormal)



print(np.dot([1.0, 1650, 3.0], rectaCosteReducido))
print(np.dot([1.0, 1650.0, 3.0], rectaEcuacionNormal))