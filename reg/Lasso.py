# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:07:36 2016

@author: shun-sa


"""
import numpy as np
from scipy import optimize as opt
from sklearn import linear_model


def _preprocess_data(X, y, fit_intercept, normalize=False, copy=True, sample_weight=None, return_mean=False):
    """
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output
        X = (X - X_offset) / X_scale
    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).
    This is here because nearly all linear models will want their data to be
    centered.
    """

    # X = check_array(X, copy=copy, accept_sparse=['csr', 'csc'],
    #                 dtype=FLOAT_DTYPES)

    if fit_intercept:
        X_offset = np.average(X, axis=0, weights=sample_weight)
        X -= X_offset
        if normalize:
            X, X_scale = f_normalize(X, axis=0, copy=False,
                                     return_norm=True)
        else:
            X_scale = np.ones(X.shape[1])

        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1])
        X_scale = np.ones(X.shape[1])
        y_offset = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


class Lasso(linear_model.LinearRegression):

    def __init__(self, alpha=0.1, iter_num=10000, rho=1., verbose=False, omega=1.5, solver="liblinear"):
        # linear_model.LinearRegression.__init__(self)
        self.alpha = alpha
        self.iter_num = iter_num
        self.rho = rho
        self.verbose = verbose
        self.omega = omega
        self.solver = solver

    def SoftMax(self, kappa, a):
        a_ = a.copy()
        a_[np.where((a_ <= kappa) & (a_ >= -kappa))] = 0
        a_[np.where(a_ > kappa)] = a_[np.where(a_ > kappa)] - kappa
        a_[np.where(a_ < - kappa)] = a_[np.where(a_ < - kappa)] + kappa
        return a_

    def solveLS(self, y, A, x):
        def residuals(x, y, A):
            error = y - np.dot(A, x)
            return np.dot(error, error.T)

        def gradient(x, y, A):
            AtA = np.dot(A.T, A)
            Aty = np.dot(A.T, y)
            return np.dot(AtA, x) - Aty

        def hessian(x, y, A):
            AtA = np.dot(A.T, A)
            return AtA
        if self.solver == "liblinear":
            return np.linalg.solve(A, y)
        elif self.solver == "Newton-CG":
            optionFlag = {'xtol': 1e-10, 'maxiter': 100, 'disp': False}
            res = opt.minimize(residuals, x, (y, A), method='Newton-CG',
                               jac=gradient, hess=hessian, options=optionFlag)
            return res.x

    def fit(self, A, b, sample_weight=None):
        """Fit linear model with coordinate descent
        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Pass directly as float64, Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
        """

        # A, b, X_offset, y_offset, X_scale = _preprocess_data(
        #     A, b, fit_intercept=self.fit_intercept, normalize=self.normalize,
        #     copy=self.copy_X, sample_weight=sample_weight)

        A_ = np.copy(A)
        b_ = np.copy(b)
        n_samples, n_features = A_.shape
        I = np.identity(n_features)
        z = np.zeros(n_features)
        u = np.zeros(n_features)
        x = np.zeros(n_features)
        MAT_ = np.dot(A_.T, A_) + (I * self.rho)
        VEC_ = np.dot(A_.T, b_)

        for iter_ in range(self.iter_num):
            z_ = z
            x = self.solveLS(VEC_ + self.rho * (z - u), MAT_, x)
            x = x * self.omega + z * (1 - self.omega)
            z = self.SoftMax(self.alpha / self.rho, x + u)
            u = u + x - z
            if self.verbose:
                print np.dot(z - z_, z - z_)

        self.coef_ = z
        self.intercept_ = np.average(b_) - np.dot(np.mean(A_, axis=0), z)
        return self

        # self.coef_=z
        # print self.coef_
        # self._set_intercept(X_offset, y_offset, X_scale)

        # self.n_iter_ = []
        # return self
