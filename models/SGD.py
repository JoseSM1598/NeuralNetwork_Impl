import numpy as np

def tensorize(x):
    return np.squeeze(np.asfarray(x))


# ####### HELPER METHODS #########
# implement Stochastic Gradient Descent to be used by our Network for training
def sgd(net, loss, T, batch_size=1, max_iter=1, learning_rate_init=1e-3,
        tol=1e-6, n_iter_no_change=10):
    N = len(T['y'])
    idx = np.argsort(np.random.random(N))
    x_scr = T['x'][idx]  # Scramble x
    y_scr = T['y'][idx]  # Scramble y

    split_x = np.split(x_scr, int(N / batch_size), axis=0)  # split data into batches of equal size
    split_y = np.split(y_scr, int(N / batch_size), axis=0)
    LT = np.zeros(max_iter)

    no_change = 0
    # Start looping through the Epochs
    for e in range(max_iter):
        step_loss = 0
        for j in range(int(N / batch_size)):
            w_0 = net.getWeights()  # Obtain current weights
            l_grads = np.zeros(len(w_0))

            for obs in range(len(split_x[j])):
                backprop = net.backprop(split_x[j][obs], split_y[j][obs], loss)
                l_grads += backprop[1]
                step_loss += backprop[0]

            dLw = l_grads / batch_size  # mean of gradients of loss
            w_0 = w_0 - learning_rate_init * dLw
            net.setWeights(w_0)  # Update weights
        LT[e] = step_loss / N  # Divide the losses acquired by dataset size

        # Check for ending
        if e > 0 and (abs(LT[e - 1] - LT[e])) < tol:
            no_change += 1
            if no_change >= n_iter_no_change:
                return (LT[:e])  # Return up until the current epoch
        else:
            no_change = 0
    return (LT)
