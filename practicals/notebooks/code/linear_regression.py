import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

def calc_yhat(X,beta):
    
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    
    yhat = np.dot(X,beta)
    
    return yhat


def lsq_weighted(X, y, d):
    """
    weighted least squares linear regression
    :param X: Input data matrix with unique values
    :param y: Target vector
    :param d: weight vector
    :return: Estimated coefficient vector for the linear regression
    """
    
    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # create weight matrix
    weights = np.mat(np.eye((len(d))))
    np.fill_diagonal(weights, d)

    # create (X^T.W.X)^-1
    Xt_W_X_inv = np.linalg.pinv(X.T.dot(weights).dot(X))

    # create (X^T.W.y)
    Xt_W_y = X.T.dot(weights).dot(y)
    
    # calculate the coefficients
    beta = np.dot(Xt_W_X_inv,Xt_W_y)

    return beta

