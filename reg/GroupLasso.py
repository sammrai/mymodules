# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:07:36 2016

@author: shun-sa


"""
import numpy as np
from scipy import optimize as opt
import time
from sklearn import linear_model
# from sklearn.linear_model.base import _preprocess_data as preprocess_data


class GroupLasso(linear_model.LinearRegression):

    def __init__(self, alpha=0.01, iter_num=10000, rho=1., verbose=False, omega=1.5, solver="liblinear"):
        # linear_model.ElasticNet.__init__(self)
        self.alpha = alpha
        self.iter_num = iter_num
        self.rho = rho
        self.verbose = verbose
        self.omega = omega
        self.solver = solver
        self.curve = []

    def SoftMax(self, kappa, a):
        # return np.sign(a) *[np.max( np.abs(i)-kappa , 0. ) for i in a]
        a_ = a.copy()
        a_[np.where((a_ <= kappa) & (a_ >= -kappa))] = 0
        a_[np.where(a_ > kappa)] = a_[np.where(a_ > kappa)] - kappa
        a_[np.where(a_ < - kappa)] = a_[np.where(a_ < - kappa)] + kappa
        return a_

    def GroupSoftMax(self, kappa, a):
        return max(0, 1 - kappa / np.linalg.norm(a)) * a

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

    def fit(self, A, b, group=None):
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
        # X, y, X_offset, y_offset, X_scale = preprocess_data(
        #     X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
        #     copy=self.copy_X, sample_weight=sample_weight)

        if group == None:
            group = [i for i in range(A.shape[1])]
        subgroup = np.array([np.array(range(len(group)))[
                            np.array(group) == i] for i in np.unique(group)])

        A_ = np.copy(A)
        b_ = np.copy(b)
        n_samples, n_features = A_.shape
        I = np.identity(n_features)
        z = np.zeros(n_features)
        u = np.zeros(n_features)
        x = np.zeros(n_features)
        MAT_ = np.dot(A_.T, A_) + (I * self.rho)
        VEC_ = np.dot(A_.T, b_)
        V = []

        start = time.time()
        for iter_ in range(self.iter_num):
            z_ = z
            x = self.solveLS(VEC_ + self.rho * (z - u), MAT_, x)
            x = x * self.omega + z * (1 - self.omega)

            for sub_g in subgroup:
                z[sub_g] = self.GroupSoftMax(
                    self.alpha / self.rho, (x + u)[sub_g])
            # z=self.SoftMax(self.alpha/self.rho,(x+u))
            u = u + x - z
            if self.verbose:
                print np.dot(z - z_, z - z_)
            self.curve.append(np.dot(z - z_, z - z_))

        self.coef_ = z
        self.intercept_ = np.average(b_) - np.dot(np.mean(A_, axis=0), z)
        # A_var = np.var(A, axis=0)
        # A_scale = np.sqrt(A_var, A_var)w
        # self._set_intercept(np.mean(A_,axis=0),np.average(b_),A_scale)
        # self.n_iter_ = []
        self.elapsed_time = time.time() - start
        return self

    # def predict(self,X):
    #     """Predict using the linear model

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix}, shape = (n_samples, n_features)
    #         Samples.
    #     Returns
    #     -------
    #     C : array, shape = (n_samples,)
    #         Returns predicted values.
    #     """
    #     X=np.array(X)

    #     if  len(X.shape)==1:
    #         return np.dot(self.coef_,X)+self.intercept_
    #     elif len(X.shape)==2:
    #         return np.array([np.dot(self.coef_,x)+self.intercept_ for x in X])
    #     else:
    #         print "##ERROR : "
    #         exit()
    # def score(self,X,y):
    #     """Returns the mean accuracy on the given test data and labels.
    #     In multi-label classification, this is the subset accuracy
    #     which is a harsh metric since you require for each sample that
    #     each label set be correctly predicted.

    #     Parameters
    #     ----------
    #     X : array-like, shape = (n_samples, n_features)
    #         Test samples.
    #     y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    #         True labels for X.

    #     Returns
    #     -------
    #     score : float
    #         Mean accuracy of self.predict(X) wrt. y.

    #     Reference
    #     -------
    #     https://en.wikipedia.org/wiki/Coefficient_of_determination

    #     """

    #     try:
    #         len(y)
    #         #from sklearn.metrics import r2_score
    #         #return r2_score(y,self.predict(X))
    #         #https://en.wikipedia.org/wiki/Coefficient_of_determination
    #         SSres = np.dot((y-self.predict(X)),(y-self.predict(X)))
    #         SStot = np.dot(y-np.mean(y),y-np.mean(y))
    #         return (1-SSres/SStot)
    #     except:
    #         raise ValueError("y should have length")
    # def count(self):
    #     def count_(A,sikii):
    #         cou=1
    #         for i in A:
    #             if np.abs(i)<sikii:
    #                 cou+=1
    #         return len(A)-cou+1
    #     return count_(self.coef_,1e-30)
