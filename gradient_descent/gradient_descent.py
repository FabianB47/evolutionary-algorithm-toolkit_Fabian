import numpy as np
import matplotlib.pyplot as plt

#Define tuction to optimize
def z_function(x, y):
    return x**2 + y**2

#Write de derivate of function
def calculate_gradient (x,y): 
    return 2*x, x*y
    
#define the domain
x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)

X, Y = np.meshgrid(x, y)

Z = z_function(X, Y)

point1 = (0.7, 0.4, z_function(0.7, 0.4))
# point2 = (0.5, 0.3, z_function(0.5, 0.3))
# point3 = (0.6, 0.1, z_function(0.6, 0.1))

learning_rate = 0.01


ax = plt.subplot(projection="3d", computed_zorder=False)

for _ in range(1000):

    #calculo el gradiente
    X_derivative, Y_derivative = calculate_gradient(point1[0], point1[1])
    X_new, Y_new= point1[0] - learning_rate * X_derivative, point1[1] - learning_rate * Y_derivative
    point1 = (X_new, Y_new, z_function(X_new, Y_new))

    ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0)
    ax.scatter(point1[0], point1[1], point1[2], color = "magenta", zorder = 1)
    plt.pause(0.001)
    ax.clear()



# #calculo el learnign rate nuevo
# a = np.array([point1_new[j]-point1_new[j-1]])
# b = np.array([a[0,0], a[0,1]])
# c = np.array([grad_new - grad])


# d = np.transpose(b)

# e = d * c

# # bt= np.transpose(b)





# # learning_rate = abs([point1_new[j]-point1_new[j-1]] * [grad_new - grad])/ (np.linalg.norm(grad_new - grad)**2)

# # #calculo la magnitud del nuevo gradiente
# # norm_gradient = np.linalg.norm(grad_new)


# print(e)






    




    # X_derivative, Y_derivative = calculate_gradient(point2[0], point2[1])
    # X_new, Y_new = point2[0] - learning_rate * X_derivative, point2[1] - learning_rate * Y_derivative
    # point2 = (X_new, Y_new, z_function(X_new, Y_new))


    # X_derivative, Y_derivative = calculate_gradient(point3[0], point3[1])
    # X_new, Y_new = point3[0] - learning_rate * X_derivative, point3[1] - learning_rate * Y_derivative
    # point3 = (X_new, Y_new, z_function(X_new, Y_new))

    # ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0)
    # ax.scatter(point1[0], point1[1], point1[2], color = "magenta", zorder = 1)
    # # ax.scatter(point2[0], point2[1], point2[2], color = "green", zorder = 1)
    # # ax.scatter(point3[0], point3[1], point3[2], color = "cyan", zorder = 1)
    # plt.pause(0.001)
    # ax.clear()













# fig = plt.figure()
# ax = plt.axes(projection='3d')

# x = np.linspace(-3,3,500)
# y = np.linspace(-3,3,500)
# X, Y = np.meshgrid(x, y)

# def z(x,y):
#     return (x**2 + y**2)
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
# ax.set_zlabel("$z$")
# fig.suptitle("$z=x²+y²$", y=1.05, fontsize=16)
# ax.plot_surface(X, Y, z(X, Y),cmap="Greens", alpha = 0.6)


# plt.show()

# # Initialization
# x = [2]  # random parameter
# y = [2]
# tol = 0.001  # epsilon
# error = 1
# h = 0.01  # h of numerical derivate calculation
# step = 0.05  # t_k
# k = 0  # count


# # Gradient Descent
# while error > tol:
#     grad_n1= (z(x[k], y[k]) - z(x[k] - h)) / h
#     x.append(x[k] - step * grad_n1)
#     grad_n= (Objective(x[k + 1])- Objective(x[k + 1]- h))/h
#     error = abs(x[k + 1] - x[k])  # to stop algorithm
#     step = abs(((x[k + 1])- x[k])*(grad_n - grad_n1))/(abs((grad_n - grad_n1))**2)
#     k += 1

#     # Plot
#     plt.plot(x[k], Objective(x[k]), 'ro')
#     plt.xlim(-15, 4)
#     plt.ylim(-50, 230)
#     plt.pause(0.1)

# plt.show()


# comentar ctr+shift+7
# # Objective function
# def Objective(a):
#     return (a + 5)**2

# # Plot
# a_values = np.linspace(-15, 4, 400)
# plt.plot(a_values, Objective(a_values), label='Objective Function')
# plt.xlabel('a')
# plt.ylabel('Objective Function Value')
# plt.title('Gradient Descent Optimization')
# plt.grid(True)
# plt.legend()


