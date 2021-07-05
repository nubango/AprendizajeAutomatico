import numpy as np
from scipy.optimize import minimize
import scipy.optimize as opt

####    COMÚN   ####
def h(x, theta):
    '''
    Calcula la predicción sobre x para un cierto theta
    '''
    return np.dot(x, theta[np.newaxis].T)

    
def hMatrix(x, theta):
    '''
    Calcula la predicción sobre x para una cierta matriz de thetas 
    '''
    return np.dot(x, theta.T)

def sigmoid(z): 
    '''
    Calcula el sigmoide del número z
    '''
    return 1 / (1 + np.exp(-z))

def sigmoidGradient(z):
    '''
    Calcula la derivada del sigmoide del número z, tanto si es un número como si es un vector/matriz
    '''
    return sigmoid(z) * (1 - sigmoid(z))

def calcula_porcentaje(Y, Z, digitsNo: int):
    '''
    Calcula el porcentaje de aciertos del entrenador
    '''

    m = Y.shape[0]

    # Creamos la matriz
    results = np.empty(m)

    # Recorremos todos los ejemplos de entrenamiento...
    for i in range (m):
        results[i] = np.argmax(Z[i])
    results = results.T

    # Vemos cuántos de ellos coinciden con Y 
    coinciden = ( Y == results )
    aciertos = np.sum(coinciden)

    # Porcentaje sobre el total de ejemplos redondeado
    return round((aciertos / m) * 100, digitsNo)

def check_accuracy(y, out):
    max_i = np.argmax(out, axis = 1) + 1
    control = (y[:, 0] == max_i) 

    # Porcentaje sobre el total de ejemplos redondeado
    return 100 * np.size(np.where(control == True)) / y.shape[0]

####    REGRESIÓN LOGÍSTICA     ####
def regularizedCost(theta, lamda: float, X, Y):
    '''
    Calcula el coste para los ejemplos, pesos y término de regularización dados
    '''
    #Variables auxiliares
    m = X.shape[0]
    H = sigmoid(np.matmul(X, theta))

    # Coste cuando Y = 1
    costeUno = np.dot(Y, np.log(H))
    # Coste cuando Y = 0
    costeCero = np.dot((1 - Y), np.log(1 - H))
    # Término de regularización (sumado al coste original)
    regTerm = lamda / (2 * m) * np.sum(theta**2) 

    return -1 / m * (costeUno + costeCero) + regTerm


def regularizedGradient(theta, lamda: float, X, Y):
    '''
    Calcula el gradiente con los ejemplos, pesos y término de regularización dados
    '''
    #Variables auxiliares
    m = X.shape[0]
    H = sigmoid(np.matmul(X, theta))

    # Cálculo del gradiente
    grad = (1/ m) * np.matmul(X.T, H-Y)

    # No queremos usar la primera componente de theta (nos la guardamos)
    aux = theta[0]
    theta[0] = 0

    # Término de regularización
    regTerm = lamda / m * theta

    #Devolvemos su valor a theta
    theta[0] = aux

    return grad + regTerm


####    REDES NEURONALES    ####
def network_cost(H, Y):
    '''
    Calcula el coste de manera vectorizada para la red neuronal,
    con una salida de la red H y la Y de los ejemplos de entrenamiento
    '''
    #Variables auxiliares
    m = Y.shape[0]

    # Usamos "multiply" en vez de "dot" para que haga multiplicación 
    # elemento a elemento, (no producto escalar)
    # y así luego los sumamos todos en vez de hacer un doble bucle
    ## Coste cuando Y = 1
    costeUno = np.multiply(Y, np.log(H)).sum() # Suma todos los 
    #elementos de la matriz (Y x H)
    ## Coste cuando Y = 0
    costeCero = np.multiply((1 - Y), np.log(1 - H)).sum() #etc

    #Coste sin regularizar
    return -1 / m * (costeUno + costeCero)


def reg_network_cost(H, Y, lamda, theta1, theta2):
    '''
    Calcula el coste (regularizado) para la red neuronal,
    con una salida de la red H y la Y de los ejemplos de entrenamiento
    '''
    #Variables auxiliares
    m = Y.shape[0]

    #Coste sin regularizar
    cost = network_cost(H, Y)

    #Término de regularización (las columnas de 1's de thetas las quitamos)
    thetaSum = ((theta1[:, 1:]**2).sum() + (theta2[:, 1:]**2).sum())
    regTerm = lamda / (2 * m) * thetaSum

    #Coste regularizado
    return (cost + regTerm)

def pesosAleatorios(L_in, L_out):
    """
    Inicializa una matriz de pesos con valores aleatorios dentro de un rango epsilon
    """
    # Rango
    epsilon = 0.12
    
    # Inicializamos la matriz con 0s
    pesos = np.zeros((L_out, 1 + L_in))
    
    # Valores aleatorios en ese intervalo
    pesos = np.random.rand(L_out, 1 + L_in) * (2 * epsilon) - epsilon
    return pesos

def forward_prop(X, theta1, theta2):
    '''
    Propagación hacia delante en la red neuronal de 2 capas
    '''
    a1 = np.hstack([np.ones([X.shape[0], 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([X.shape[0], 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def back_prop (nn_params, num_entradas, num_ocultas, num_etiquetas, X, y, lamda):
    """
    Implementa la propagación hacia atrás de la red neuronal con 2 capas
    Tenemos que convertir el vector "nn_params" en 2 matrices, ya que viene 
    desenrollado. 

    Devuelve el coste y el vector de gradientes (desenrollado también)
    """
    # 1. Volvemos a construir las matrices de pesos
    theta1 = np.reshape(nn_params[:num_ocultas * (num_entradas + 1)],
                        (num_ocultas, num_entradas + 1)) # (25,401)
    theta2 = np.reshape(nn_params[num_ocultas * (num_entradas + 1):],
                        (num_etiquetas, num_ocultas + 1)) # (10,26)

    # Número de ejemplos de entrenamiento
    m = X.shape[0]
    X = np.hstack([np.ones([m, 1]), X]) #Para el término indep.

    # 2. Hacemos la propagación hacia delante para obtener las activaciones
    a1, z2, a2, z3, h = forward_prop(X, theta1, theta2) 

    # 3. Inicializamos las matrices delta (con ceros)
    delta1 = np.zeros((num_ocultas, num_entradas + 1))
    delta2 = np.zeros((num_etiquetas, num_ocultas + 1))
    
    # 4. RETRO - PROPAGACIÓN
    for t in range(m):
        a1t = a1[t, :] # (1, 401)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :] # (1, 10)
        yt = y[t] # (1, 10)

        #Error en la capa de salida
        d3t = ht - yt
        #Error en la capa oculta
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    # 5. Calculamos el coste regularizado
    regCost = reg_network_cost(h, y, lamda, theta1, theta2)
    
    # 6. Calculamos el gradiente...
    delta1 = delta1 / m
    delta2 = delta2 / m

    # ... y lo regularizamos
    delta1[:,1:] = delta1[:,1:] + (lamda / m) * theta1[:,1:]
    delta2[:,1:] = delta2[:,1:] + (lamda / m) * theta2[:,1:]
    
    # Desenrollamos el gradiente y lo devolvemos junto al coste
    grad = np.concatenate((delta1.ravel(), delta2.ravel()))
    return regCost, grad

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

def trainNeutralNetwork(num_entradas, num_ocultas, num_etiquetas, X, y, lamda, num_iter):
    '''
    Entrena una red neuronal de 2 capas y devuelve las matrices de pesos para cada capa
    La y debe estar en formato onehot
    '''
    # 1. Comenzamos con unos pesos aleatorios
    theta1 = pesosAleatorios(num_entradas, num_ocultas)
    theta2 = pesosAleatorios(num_ocultas, num_etiquetas)
    nn_params = np.concatenate((theta1.ravel(), theta2.ravel())) #Los unimos en 1 solo vector

    # 2. Llamamos a la función minimize para obtener las matrices de pesos óptimos
    # (las que hacen que haya un mínimo en el coste devuelto, usando back_prop)
    thetaOpt = minimize(fun=back_prop,
                       x0=nn_params,
                       args=(num_entradas,
                             num_ocultas,
                             num_etiquetas,
                             X, y, lamda),
                       method='TNC',
                       jac=True,
                       options={'maxiter':num_iter}).x

    # 3. Tenemos que reconstruir los pesos a partir del vector
    theta1 = np.reshape(thetaOpt[:num_ocultas * (num_entradas + 1)],
                        (num_ocultas, num_entradas + 1)) 
    theta2 = np.reshape(thetaOpt[num_ocultas * (num_entradas + 1):],
                        (num_etiquetas, num_ocultas + 1)) 

    # Devolvemos los pesos óptimos
    return [theta1, theta2]

def optimize(backprop, params_rn, input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda, num_iter):
    result = opt.minimize(fun=backprop, x0=params_rn,
    args=(input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda),
    method='TNC', jac=True, options={'maxiter': num_iter})
    return result.x

def neural_network_training(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda, num_iter):
    Epsilon = 0.12

    # Init Neural Network params
    theta1 = pesosAleatorios(theta1.shape[1] - 1, theta1.shape[0])
    theta2 = pesosAleatorios(theta2.shape[1] - 1, theta2.shape[0])

    # Train Neural Network
    params_rn = np.concatenate([theta1.reshape(-1), theta2.reshape(-1)])
    theta_opt = optimize(back_propagation, params_rn, input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda, num_iter)
    theta1_opt = np.reshape(theta_opt[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)))
    theta2_opt = np.reshape(theta_opt[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1 )))

    return check_accuracy(Y, forward_prop(X, theta1_opt, theta2_opt)[4])