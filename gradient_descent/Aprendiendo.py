import numpy as np
import matplotlib.pyplot as plt

#Define tuction to optimize
def z_function(x, y):
    return np.sin(5*x) * np.cos(5* y) / 5

#Write de derivate of function
def calculate_gradient (x,y): 
    return np.cos(5 * x) * np.cos(5 * y), -np.sin(5 * x) * np.sin(5 * y)
    
#define the domain
x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)

X, Y = np.meshgrid(x, y)

Z = z_function(X, Y)

point1 = np.array([[0.7, 0.4, z_function(0.7, 0.4)]])
# point2 = (0.5, 0.3, z_function(0.5, 0.3))
# point3 = (0.6, 0.1, z_function(0.6, 0.1))

learning_rate = 0.01
j = 0
norm_gradient = 1
tol = 0.0000001


ax = plt.subplot(projection="3d", computed_zorder=False)

while norm_gradient > tol:
    #calculo el gradiente
    X_derivative, Y_derivative = calculate_gradient(point1[j,0], point1[j,1])
    grad = np.array([[X_derivative,Y_derivative]])

    #calculo el siguiente punto
    X_new, Y_new= point1[0,0] - learning_rate * X_derivative, point1[j,1] - learning_rate * Y_derivative
    point1= np.append(point1,[[X_new, Y_new, z_function(X_new, Y_new)]],axis=0)

    #calculo el gradiente del siguiente punto
    j +=1
    X_derivative_new, Y_derivative_new = calculate_gradient(point1[j,0], point1[j,1])
    grad_new= np.array([[X_derivative_new,Y_derivative_new]])

    #calculo la normal del nuevo gradiente
    norm_gradient = np.linalg.norm(grad_new) 

    ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0)
    ax.scatter(point1[j,0], point1[j,1], point1[j,2], color = "magenta", zorder = 1)
    plt.pause(0.001)
    ax.clear()

    print(norm_gradient)












