import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
def display2d():
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    capped_valid = q_valid_data
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1) + 1/2**res) / (2 ** res - 1) * max_val
    from matplotlib import pyplot as plt
    numbers = tf.abs(tf.reshape(capped_valid, [1000*K*M,])).numpy()
    plt.hist(numbers, bins=2**4, label="4 bits quantization")
    plt.hist(numbers, bins=2**5, label="5 bits quantization")
    plt.hist(numbers, bins=2**6, label="6 bits quantization")
    plt.hist(numbers, bins=2**8, label="8 bits quantization")
    plt.title("Effect of Quantization On channel gains")
    plt.legend()
    plt.show()
def compare_model_with_greedy_under_partial_feedback(Nrf):
    for Nrf in range(Nrf, Nrf + 1):
        # add or remove points using x and y
        x = np.arange(1, 64)  # links
        y = np.arange(1, 32)  # bits
        grid = np.load("partial_feedback_and_DNN_scheduler.npy")
        out = np.zeros((128,))
        out_x = []
        out_y = []
        for i in x:
            for j in y:
                if grid[i, j].any() != 0:
                    out[i * (6 + j)] = max(grid[i - 1, j - 1, Nrf - 1], out[i * (6 + j)])

        for i in range(0, 128):
            if out[i] != 0:
                out_x.append(i)
                out_y.append(out[i])
        plt.plot(np.array(out_x), np.array(out_y), label="DNN Nrf={}".format(Nrf))
        grid = np.load("grid_search_all_under128.npy")
        # add or remove points using x and y
        x = np.arange(1, 64)  # links
        y = np.arange(1, 32)  # bits
        out = np.zeros((128,))
        out_x = []
        out_y = []
        for i in x:
            for j in y:
                if grid[i, j].any() != 0:
                    out[i * (6 + j)] = max(-grid[i - 1, j - 1, Nrf - 1], out[i * (6 + j)])
        for i in range(0, 128):
            if out[i] != 0:
                out_x.append(i)
                out_y.append(out[i])

        plt.plot(np.array(out_x), np.array(out_y), label="greedy Nrf={}".format(Nrf))
    plt.legend()
    plt.xlabel("bits per user")
    plt.ylabel("sum rate")
    plt.show()
def get_hull(x, y):
    new_x = []
    new_y = []
    x_0 = x[0]
    y_0 = y[0]
    new_x.append(x_0)
    new_y.append(y_0)
    counter = 1
    while True:
        if counter >= len(x)-1:
            break
        max_slope = -10000
        max_index = counter
        for i in range(counter, len(x)):
            slope = (y[i]-y[counter-1])/(x[i]-x[counter-1])
            if slope > max_slope:
                max_slope = slope
                max_index = i
        if max_slope >= 0:
        # if True:
            new_x.append(x[max_index])
            new_y.append(y[max_index])
        new_counter = max_index + 1
        if new_counter == counter:
            break
        counter = new_counter

    return new_x, new_y
def compare_model_with_greedy_under_partial_feedback_outer_points(Nrf):
    greedy_up = [13.77, 23.26, 31.25, 38.16, 44.19, 49.41, 53.97, 57.97]
    gumbel_up = [13.76, 21.94, 29.66, 36.48, 42.02, 47.1, 52.01, 55.90]
    plot_perfect_CSI = True
    for Nrf in range(Nrf, Nrf + 1):
        # add or remove points using x and y
        x = np.arange(1, 64)  # links
        y = np.arange(1, 32)  # bits
        grid = np.load("partial_feedback_and_DNN_scheduler.npy")
        out = np.zeros((128,))
        out_x = []
        out_y = []
        for i in x:
            for j in y:
                if grid[i, j].any() != 0:
                    out[i * (6 + j)] = max(grid[i - 1, j - 1, Nrf - 1], out[i * (6 + j)])

        for i in range(0, 128):
            if out[i] != 0:
                out_x.append(i)
                out_y.append(out[i])
        out_x, out_y = get_hull(out_x, out_y)
        plt.plot(np.array(out_x), np.array(out_y), label="DNN Nrf={}".format(Nrf))
        if plot_perfect_CSI:
            plt.plot(np.array([1, max(out_x)]), np.array([gumbel_up[Nrf - 1], gumbel_up[Nrf - 1]]), label="gumbel full CSI")
        grid = np.load("grid_search_all_under128.npy")
        # add or remove points using x and y
        x = np.arange(1, 64)  # links
        y = np.arange(1, 32)  # bits
        out = np.zeros((128,))
        out_x = []
        out_y = []
        for i in x:
            for j in y:
                if grid[i, j].any() != 0:
                    out[i * (6 + j)] = max(-grid[i - 1, j - 1, Nrf - 1], out[i * (6 + j)])
        for i in range(0, 128):
            if out[i] != 0:
                out_x.append(i)
                out_y.append(out[i])
        out_x, out_y = get_hull(out_x, out_y)
        plt.plot(np.array(out_x), np.array(out_y), label="greedy Nrf={}".format(Nrf))
        if plot_perfect_CSI:
            plt.plot(np.array([1, max(out_x)]), np.array([greedy_up[Nrf-1], greedy_up[Nrf-1]]), label="greedy full CSI")
    plt.legend()
    plt.xlabel("bits per user")
    plt.ylabel("sum rate")
    plt.show()
if __name__ == "__main__":

    compare_model_with_greedy_under_partial_feedback_outer_points(4)
    A[2]
    Nrf = 8
    for Nrf in range(Nrf, Nrf+1):
        # add or remove points using x and y
        x = np.arange(1, 65)  # links
        y = np.arange(1, 33)  # bits
        grid = np.load("partial_feedback_and_DNN_scheduler.npy")
        out = np.zeros((128,))
        out_x = []
        out_y = []
        for i in x:
            if grid[i-1, 0].any() != 0:
                out_x = []
                out_y = []
                for j in y:
                    if grid[i - 1, j - 1, Nrf - 1] != 0:
                        out_x.append(i*(j+6))
                        out_y.append(grid[i - 1, j - 1, Nrf - 1])
                plt.plot(np.array(out_x), np.array(out_y), "o", label="{} links".format(i))
        plt.legend()
        plt.show()
        A[2]
        for i in range(0, 128):
            if out[i] != 0:
                out_x.append(i)
                out_y.append(out[i])
        plt.plot(np.array(out_x), np.array(out_y), label="DNN Nrf={}".format(Nrf))
        grid = np.load("grid_search_all_under128.npy")
        # add or remove points using x and y
        x = np.arange(1, 64)  # links
        y = np.arange(1, 32)  # bits
        out = np.zeros((128, ))
        out_x = []
        out_y = []
        for i in x:
            for j in y:
                if grid[i, j].any() != 0:
                    out[i*(6+j)] = max(-grid[i-1,j-1,Nrf-1], out[i*(6+j)])
        for i in range(0, 128):
            if out[i] != 0:
                out_x.append(i)
                out_y.append(out[i])

        plt.plot(np.array(out_x), np.array(out_y), label="greedy Nrf={}".format(Nrf))
    plt.legend()
    plt.xlabel("bits per user")
    plt.ylabel("sum rate")
    plt.show()