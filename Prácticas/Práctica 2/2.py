import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from pandas.io.parsers import read_csv
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    return valores.astype(float)

def g(z):
    return 1/(1+np.exp(-z))

def coste(Theta, X, Y):
	m = np.shape(X)[0]
	Aux = (np.log(g(X @ Theta))).T @ Y
	Aux += (np.log(1 - g(X @ Theta))).T @ (1 - Y)
	return -Aux / m

def gradiente(Theta, X, Y):
	m = np.shape(X)[0]
	Aux = X.T @ (g(X @ Theta) - Y)
	return Aux / m

def pinta_frontera_recta(Theta, X, Y):
	plt.figure()

	pos = np.where(Y == 1)
	neg = np.where(Y == 0)
	plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+')
	plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='o')

	x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
	x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
	
	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
						   np.linspace(x2_min, x2_max))

	h = g(np.c_[np.ones((xx1.ravel().shape[0], 1)),
				xx1.ravel(),
				xx2.ravel()].dot(Theta))


	h = h.reshape(xx1.shape)

	plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    
	plt.legend(('Admitted', 'Not Admitted'), loc='upper right')
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
    
	plt.show()


def accuracy_percentage(Theta, X, Y):

	Z = g(X@Theta)
	pos = np.where(Y == 1)
	neg = np.where(Y == 0)
	z_pos = np.where(Z >= 0.5)
	z_neg = np.where(Z < 0.5)

	pos_perc = np.shape(z_pos)[1] * 100 / np.shape(pos)[1]
	neg_perc = np.shape(z_neg)[1] * 100 / np.shape(neg)[1]	
	if pos_perc > 100: pos_perc = 200 - pos_perc
	if neg_perc > 100: neg_perc = 200 - neg_perc
	
	print("{}% de positivos clasificado correctamente.".format(pos_perc))
	print("{}% de positivos clasificado correctamente.".format(neg_perc))

def parte1():

	data = carga_csv("ex2data1.csv")
	X = data[:, :-1]
	Y = data[:, -1]

	m = np.shape(X)[0]
	Xvec = np.hstack([np.ones([m, 1]), X])
	n = np.shape(Xvec)[1]

	Theta = np.zeros(n)
	print(coste(Theta, Xvec, Y))
	print(gradiente(Theta,Xvec, Y))

	result = opt.fmin_tnc(func=coste , x0=Theta , fprime=gradiente , args=(Xvec, Y))
	theta_opt = result [0]

	pinta_frontera_recta(theta_opt, X, Y)

	accuracy_percentage(theta_opt, Xvec, Y)

def coste_reg(Theta, X, Y, Lambda):
    m = np.shape(X)[0]
    Aux = (np.log(g(X @ Theta))).T @ Y
    Aux += (np.log(1 - g(X @ Theta))).T @ (1 - Y)
    Cost = -Aux / m
    Regcost = (Lambda / (2 * m)) * sum(Theta ** 2)
    return Cost + Regcost 
	
def gradiente_reg(Theta, X, Y, Lambda):
    m = np.shape(X)[0]
    Aux = X.T @ (g(X @ Theta) - Y)
    Grad = Aux / m
    theta_aux = Theta
    theta_aux[0] = 0.0
    Grad = Grad + (Lambda / m) * theta_aux
    return Grad

def plot_decisionboundary(Theta, X, Y, poly):
    plt.figure()

    pos = np.where(Y == 1)
    neg = np.where(Y == 0)
    pts_pos = plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+')
    pts_neg = plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='o')

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), 
                           np.linspace(x2_min, x2_max))

    h = g(poly.fit_transform(np.c_[xx1.ravel(),
                                   xx2.ravel()]).dot(Theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.show()

def parte2():
	data = carga_csv("ex2data2.csv")

	X = data[:, :-1]
	Y = data[:, -1]

	m = np.shape(X)[0]
	poly = PolynomialFeatures(degree=6)
	X_reg = poly.fit_transform(X)
	n = np.shape(X_reg)[1]

	Lambda = 0.0005
	Theta = np.zeros(n)
	print(coste_reg(Theta, X_reg, Y, Lambda))

	result = opt.fmin_tnc(func=coste_reg, x0=Theta,
	                      fprime=gradiente_reg, args=(X_reg, Y, Lambda))
	theta_opt = result[0]

	plot_decisionboundary(theta_opt, X, Y, poly)

#parte1()
parte2()