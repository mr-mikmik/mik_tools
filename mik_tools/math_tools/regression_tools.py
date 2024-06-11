import numpy as np


# fit some polynomial

def polynomial_expansion(X, polynomial_degrees=None, polynomial_degree=None):
    """
    compute the polynomial expansion of x
    :param X: numpy array of shape (N,)
    :param polynomial_degrees:
    :param polynomial_degree:
    :return: X_poly_expansion: numpy array of shape (N, polynomial_degree+1)
    """
    # compoes a
    # create the polynomial
    if polynomial_degrees is None:
        polynomial_degrees = np.arange(polynomial_degree + 1)
    if polynomial_degree is None and polynomial_degrees is None:
        raise NotImplementedError('Need to provide either polynomial_coeffs or polynomial_degree')
    # compute the polynomial
    X_poly = np.zeros((X.shape[0], polynomial_degrees.shape[0]))
    for i, coeff in enumerate(polynomial_degrees):
        X_poly[:, i] = X ** coeff
    return X_poly


def polynomial_least_squares_regression(x_values, y_values, polynomial_degrees=None, polynomial_degree=None):
    """
    compute the polynomail
    :param x_values: numpy array of x values of shape (N,)
    :param y_values: numpy array of y values of shape (N,)
    :param polynomial_coeffs:
    :param polynomial_degree:
    :return:
    """
    # compoes a
    # create the polynomial
    X_poly = polynomial_expansion(x_values, polynomial_degrees=polynomial_degrees, polynomial_degree=polynomial_degree) # (N, polynomial_degree+1)
    # compute the least squares solution
    # This computes polynomial_coeffs such that X_poly @ polynomial_coeffs = y_values
    # polynomial_coeffs = (X_poly.T @ X_poly)^-1 @ X_poly @ y_values
    polynomial_coeffs = np.linalg.lstsq(X_poly, y_values, rcond=None)[0]
    return polynomial_coeffs



