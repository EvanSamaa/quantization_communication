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
    grid = np.load("grid_search_all_under128.npy")
    fig = plt.figure()
    # add or remove points using x and y
    x = np.arange(1, 64)  # links
    y = np.arange(1, 32)  # bits
    Nrf = 8
    out = np.zeros((128, ))
    for i in x:
        new_curve = []
        new_val = []
        for j in y:
            if grid[i, j].any() != 0:
                new_curve.append(i*(6+j))
                out[i*(6+j)] = max(out[i*(6+j)], -grid[i-1,j-1,Nrf-1])
    plt.plot(np.arange(128, ), out, label="{} links".format(i))
    plt.legend()
    plt.show()