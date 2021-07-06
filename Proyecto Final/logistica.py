import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
pd.options.mode.chained_assignment = None
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
import os

# Funcion Sigmoide
def sigmoid(Z):
	return 1/(1+np.exp(-Z))

# Funcion de Coste
def cost(O, X, Y):
	return -((np.log(sigmoid(X.dot(O)))).T.dot(Y) + (np.log(1-sigmoid(X.dot(O)))).T.dot(1-Y))/X.shape[0]

# Operacion que hace el gradiente, devuelve un vector de valores
def gradient(O, X, Y):  
	return (X.T.dot(sigmoid(X.dot(O))-Y))/X.shape[0]

# Determina el porcentaje de aciertos comparando los resultados estimados con los resultados reales
def success_percentage(theta, X, Y):
	Z = sigmoid(X@theta)
	pos = np.where(Y == 1)
	neg = np.where(Y == 0)
	z_pos = np.where(Z >= 0.5)
	z_neg = np.where(Z < 0.5)

	pos_perc = np.shape(z_pos)[1] * 100 / np.shape(pos)[1]
	neg_perc = np.shape(z_neg)[1] * 100 / np.shape(neg)[1]	
	if pos_perc > 100: pos_perc = 200 - pos_perc
	if neg_perc > 100: neg_perc = 200 - neg_perc
	
	print("Positives accuracy: " + format(pos_perc))
	print("Negatives accuracy: " + format(neg_perc))



def logistic_regression(X, y):	

	m = np.shape(X)[0]
	X_ones = np.hstack([np.ones([m, 1]), X])
	n = np.shape(X_ones)[1]

	Theta = np.zeros(n)

	result = opt.fmin_tnc(func=cost, x0=Theta, fprime=gradient, args=(X_ones, y))
	theta_opt = result[0]
	
	success_percentage(theta_opt, X_ones, y)
	print("\n")


df = pd.read_csv("abalone_original.csv")
    
df['newRings'] = np.where(df['rings'] > 10,1,0)

X = df.drop(['newRings','rings','sex'], axis = 1)
y = df['newRings']

logistic_regression(X ,y)