import numpy as np

def conv_function(a_temp, W, b):

    s = a_temp*W
    Z = np.sum(s)
    Z = float(Z + b)

    return Z

def conv_layer(A_prev, W, b):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    n_H = int(n_H_prev - f) + 1
    n_W = int(n_W_prev - f) + 1

    #initialize output Z

    z = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(n_H):

            window_h_start = h
            window_h_end = h + f

            for w in range(n_W):

                window_w_start = w
                window_w_end = w + f

                for c in range(n_C):

                    a_temp = a_prev[window_h_start:window_h_end, window_w_start:window_w_end, :]

                    Z[i, h, w, c] = conv_function(a_temp, W[:, :, :, c], b[:,:,:,c])

    return Z












