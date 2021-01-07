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
if __name__ == "__main__":
    withrecond = "../../../trained_models/better_quantizer/with_recon.npy"
    withoutrecond = "../../../trained_models/better_quantizer/without_recon.npy"
    withrecond = np.load(withrecond)
    withoutrecond = np.load(withoutrecond)
    plt.plot(withrecond[:,0], label="with reconstruction loss, threshold")
    plt.plot(withoutrecond[:,0], label="without reconstruction loss, threshold")
    plt.plot(withrecond[:,1], label="with reconstruction loss, sm")
    plt.plot(withoutrecond[:,1], label="without reconstruction loss, sm")
    plt.legend()
    plt.show()

    grid = np.load("grid_search_all_under128.npy")
    fig = plt.figure()
    # add or remove points using x and y
    x = np.arange(1, 64)  # links
    y = np.arange(1, 32)  # bits
    Nrf = 5
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

    plt.plot(np.array(out_x), np.array(out_y), label="{} links".format(i))
    plt.legend()
    plt.show()