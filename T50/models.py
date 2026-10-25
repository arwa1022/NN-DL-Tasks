import numpy as np

def SL_Perceptron(X, Y, n_epochs, W, L, b=None):
    # X and Y should be np arrays
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    W = np.asarray(W, dtype=float)

    n_samples = len(X)

    if b is not None:
        X = np.column_stack((np.ones(n_samples), X))
        W = np.insert(W, 0, b)
        
    for i in range(n_epochs):
        W_old = W.copy()
        for j in range(n_samples):
            net = np.dot(W.T, X[j, :])
            Y_pred = 1 if net >= 0 else -1
            if Y_pred != Y[j]:
                error = Y[j] - Y_pred
                W = W + L * error * X[j, :]
                print(f"Weight for epoch {i} row {j} = {W}")

        if np.array_equal(W_old, W):
            print("Number of Epochs needed for convergence:", i + 1)
            break

    return W # w0 is bias followed by Weights if bias exit


def adaline(Xi, Ti, L, epochs, Wi, mse_threshold  ,b=None ):
    n_sample = len(Xi)
    for i in range(epochs):
        errors = 0  # sum all squared errors
        for j in range(n_sample):
            if b is not None:
                v = np.dot(Wi, Xi[j]) + b
            else:
                v = np.dot(Wi, Xi[j])
            Yi = v  
            error = Ti[j] - Yi
            errors += (error ** 2.00)
            Wi = Wi + (L * error * Xi[j])
            if b is not None:
                b = b + (L * error * 1) # X0
        mse = (1 / n_sample) * errors
        if mse < mse_threshold:
            print(f"Training stopped at epoch {i+1} with MSE: {mse}")
            break

    if b is not None:
        return Wi, b

    return Wi