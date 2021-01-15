import numpy as np
import matplotlib.pyplot as plt
def display2d():
    grid = np.load("grid_search.npy")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # add or remove points using x and y
    x = np.array([1, 2, 5, 10, 64]) # links
    y = np.array([1, 2, 4, 8, 16, 32]) # bits
    X, Y = np.meshgrid(x, y)
    print(Y)
    zs = np.zeros((len(x), len(y)), dtype=np.float32)
    # delete the for looop to display only one Nrf
    #
    for N in range(7,8):
        for i in range(len(x)):
            for j in range(len(y)):
                zs[i, j] = -grid[x[i] - 1, y[j] - 1, N]
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z)
    ax.set_xlabel('links fed back')
    ax.set_ylabel('bits')
    ax.set_zlabel('SR')

    plt.show()
if __name__ == "__main__":
    grid = np.load("grid_search.npy")
    fig = plt.figure()
    # add or remove points using x and y
    x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])  # links
    y = np.arange(1,33)  # bits
    Nrf = 5
    curves = []
    vals = []
    for i in x:
        new_curve = []
        new_val = []
        for j in y:
            new_curve.append(i*(6+j))
            new_val.append(-grid[i-1,j-1,Nrf-1])
        curves.append(np.array(new_curve))
        vals.append(np.array(new_val))
        plt.plot(np.array(new_curve), np.array(new_val), label="{} links".format(i))
    plt.legend()
    plt.show()