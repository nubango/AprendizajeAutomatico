
#leemos el contenido del archivo csv y lo convertimos en un array de numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from  pandas . io . parsers  import read_csv

def carga_csv(file):
    valores = read_csv ( file ,  header=None) . to_numpy ()
    return valores.astype(float)

datos = carga_csv("ex1data1.csv")
X = datos[:, 0]    
Y = datos[:, 1]
m = len(X)    
alpha = 0.01   
#inicializamos theta0 y theta1 a 0 
theta_0 = theta_1 = 0
#subrayado para que no se defina ninguna variable nueva
for _ in range(1500):        
    sum_0 = sum_1 = 0
    for i in range(m):            
        sum_0 +=  (theta_0 + theta_1 * X[i]) - Y[i]            
        sum_1 += ((theta_0 + theta_1 * X[i]) - Y[i]) * X[i]        
    theta_0 = theta_0 - (alpha / m) * sum_0        
    theta_1 = theta_1 - (alpha / m) * sum_1

#grafica recta con menos coste a raiz de encontrar theta0 y theta1
plt.plot(X, Y, "x")    
min_x = min(X)    
max_x = max(X)   
min_y = theta_0 + theta_1 * min_x    
max_y = theta_0 + theta_1 * max_x    
plt.plot([min_x, max_x], [min_y, max_y])    
plt.savefig("resultado.pdf")

def h(x, Theta):
    return Theta[0] + Theta [1]*x

#funcion de coste
def coste(X,Y,Theta):
    m = np.shape(X)[0]
    return 1/(2*m) * np.sum((h(X, Theta) - Y)**2)

#metodo calculo de las matrices para la grafica 3D
def make_data(t0_range,t1_range,X,Y):
    """GeneralasmatricesX,Y,Zparagenerarunploten3D"""
    step=0.1
    Theta0=np.arange(t0_range[0],t0_range[1],step)
    Theta1=np.arange(t1_range[0],t1_range[1],step)

    Theta0,Theta1=np.meshgrid(Theta0,Theta1)
    #Theta0yTheta1tienenlasmismadimensiones,deformaque
    # #cogiendounelementodecadaunosegeneranlascoordenadasx,y
    # #detodoslospuntosdelarejilla

    Coste=np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix,iy]= coste(X,Y,[Theta0[ix,iy],Theta1[ix,iy]])

    return Theta0,Theta1,Coste

#graficas 3D
fig = plt.figure()
ax= fig.gca(projection = '3d')  #ax=Axes3D(fig)

A, B, Z = make_data([-5, 5], [-5, 5], X, Y)

surf = ax.plot_surface(A, B, Z, cmap=cm.coolwarm, linewidth = 0, antialiased = False)

A, B, Z = make_data([-10, 8.5], [-1, 4], X, Y)

fig = plt.figure()
plt.contour(A,B,Z, np.logspace(-2,3,20), cmap=cm.coolwarm)
plt.scatter(theta_0, theta_1, marker='X')

plt.show()

