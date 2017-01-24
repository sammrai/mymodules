# coding:utf-8


# http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/logit.pdf
# http://statweb.stanford.edu/~tibs/ElemStatLearn/
# web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

import numpy as np
from scipy import optimize
from sklearn import linear_model
import reg


class RobustLinearRegression(linear_model.LogisticRegression):
    """LogisticRegression inprementation using ADMM"""

    def __init__(self, solver='Newton-CG', iter_max=100):
        linear_model.LogisticRegression.__init__(self)
        self.solver = solver
        self.iter_max = iter_max

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cal_theta(self, theta, X, y):
        m = float(len(y))

        def safe_log(x, minval=0.0000000001):
            return np.log(x.clip(min=minval))

        def J(theta, *args):
            X, y = args
            h = self.sigmoid(np.dot(X, theta))
            return 1 / m * np.sum(-y * safe_log(h) - (1 - y) * safe_log(1 - h))

        def grad_J(theta, *args):
            X, y = args
            h = self.sigmoid(np.dot(X, theta))
            # print "grad: ", 1/m*np.dot(X.T,h-y)
            return 1 / m * np.dot(X.T, h - y)

        def hess_J(theta, *args):
            X, y = args
            h = self.sigmoid(np.dot(X, theta))
            X_c = np.array([i * (1 - i) * j for i, j in zip(h, X)])
            # print "hess: ",1/m*np.dot(X.T,X_c)
            return 1 / m * np.dot(X.T, X_c)
        # return theta
        optionFlag = {'maxiter': self.iter_max, 'disp': False}
        res = optimize.minimize(J, theta, method=self.solver,
                                jac=grad_J, hess=hess_J, options=optionFlag, args=(X, y))
        return res.x

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_ = np.copy(X)
        y_ = np.copy(y)
        X_ = np.hstack((np.ones((len(y), 1)), X_))

        n_samples, n_features = X_.shape
        theta = np.zeros(n_features)

        theta = self.cal_theta(theta, X_, y_)
        # self.coef_=theta
        self.coef_ = theta[1:][np.newaxis, :]
        self.intercept_ = theta[0][np.newaxis]


class SparseLogisticRegression(linear_model.LogisticRegression):
    """SparseLogisticRegression inprementation using ADMM"""

    def __init__(self, iter_admm=100, solver='Newton-CG', iter_max=100, omega=1.5, alpha=0.1, rho=1.):
        linear_model.LogisticRegression.__init__(self)
        self.solver = solver
        self.iter_max = iter_max
        self.iter_admm = iter_admm
        self.alpha = alpha
        self.omega = omega
        self.rho = rho
        self.curve = []

    def SoftMax(self, kappa, a):
        # return np.sign(a)*max(np.abs(x)-kappa,0)
        a_ = a.copy()
        a_[np.where((a_ <= kappa) & (a_ >= -kappa))] = 0
        a_[np.where(a_ > kappa)] = a_[np.where(a_ > kappa)] - kappa
        a_[np.where(a_ < - kappa)] = a_[np.where(a_ < - kappa)] + kappa
        return a_

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cal_theta(self, theta, X, y, u_z):
        m = float(len(y))

        def safe_log(x, minval=0.0000000001):
            return np.log(x.clip(min=minval))

        def J(theta, *args):
            X, y = args
            h = self.sigmoid(np.dot(X, theta))
            return 1 / m * (np.sum(-y * safe_log(h) - (1 - y) * safe_log(1 - h))) + self.rho / 2. * np.dot((theta + u_z).T, theta + u_z)

        def grad_J(theta, *args):
            X, y = args
            h = self.sigmoid(np.dot(X, theta))
            return 1 / m * np.dot(X.T, h - y) + self.rho * (theta + u_z)

        def hess_J(theta, *args):
            X, y = args
            h = self.sigmoid(np.dot(X, theta))
            X_c = [i * (1 - i) * j for i, j in zip(h, X)]
            return 1 / m * np.dot(X.T, X_c) + np.diag([self.rho for i in range(self.n_features)])
        # return theta
        optionFlag = {'maxiter': self.iter_max, 'disp': False}
        res = optimize.minimize(J, theta, method=self.solver,
                                jac=grad_J, hess=hess_J, options=optionFlag, args=(X, y))
        return res.x

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Class >3 does not support.")
        X_ = np.copy(X)
        y_ = np.copy(y)
        X_ = np.hstack((np.ones((len(y), 1)), X_))
        self.n_samples, self.n_features = X_.shape
        theta = np.zeros(self.n_features)
        z = np.zeros(self.n_features)
        u = np.zeros(self.n_features)

        for i in range(self.iter_admm):
            # z_=z
            theta = self.cal_theta(theta, X_, y_, u - z)
            theta = theta * self.omega + z * (1 - self.omega)
            z[1:] = self.SoftMax(self.alpha / self.rho, (theta + u)[1:])
            z[0] = (theta + u)[0]
            u = u + theta - z
            diff = np.linalg.norm(z - theta) / np.linalg.norm(theta)
            self.curve.append(diff)

        self.coef_ = z[1:][np.newaxis, :]
        self.intercept_ = z[0][np.newaxis]
