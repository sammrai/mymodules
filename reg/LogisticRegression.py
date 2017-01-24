# coding:utf-8


# http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/logit.pdf
# http://statweb.stanford.edu/~tibs/ElemStatLearn/
# web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

import numpy as np
from scipy import optimize
from sklearn import linear_model
import numpy as np
# import matplotlib.pyplot as plt
import st


class LogisticRegression_ADMM(linear_model.LogisticRegression):
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
        print "LogisticRegression"

        self.classes_ = np.unique(y)
        X_ = np.copy(X)
        y_ = np.copy(y)
        X_ = np.hstack((np.ones((len(y), 1)), X_))

        n_samples, n_features = X_.shape
        theta = np.zeros(n_features)

        theta = self.cal_theta(theta, X_, y_)
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


class LogisticRegression(linear_model.LogisticRegression):
    """Multiclass Logistic Regression inprementation using Newton-Raphson algorithm"""

    def __init__(self, iter_max=20, penalty=None, alpha=1., rho=1., iter_admm=300, verbose=False, error=1e-4):
        linear_model.LogisticRegression.__init__(self)
        self.iter_admm = int(iter_admm)
        self.iter_max = int(iter_max)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.penalty = penalty
        self.curve = []
        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            st.mkdir(".temp", rm=True)
        self.error = error

    def SoftMax(self, kappa, a):
        # return np.sign(a)*max(np.abs(x)-kappa,0)
        a_ = a.copy()
        a_[np.where((a_ <= kappa) & (a_ >= -kappa))] = 0
        a_[np.where(a_ > kappa)] = a_[np.where(a_ > kappa)] - kappa
        a_[np.where(a_ < - kappa)] = a_[np.where(a_ < - kappa)] + kappa
        return a_

    def GroupSoftMax(self, kappa, a):
        # if self.verbose:print np.linalg.norm(a)
        return max(0, 1 - kappa / np.linalg.norm(a)) * a

    def cal_theta(self, theta, X, y, u_z=None, l1=False):
        def safe_log(x, minval=0.0000000001):
            return np.log(x.clip(min=minval))

        def calp(theta, X):
            Y = np.zeros((self.n_samples, self.K))
            for n in range(self.n_samples):
                denominator = 0.0
                for k in range(self.K):
                    denominator += np.exp(np.dot(theta[:, k], X[n, :]))
                for k in range(self.K):
                    Y[n, k] = np.exp(
                        np.dot(theta[:, k], X[n, :])) / denominator
            return Y

        def J(theta, X, Y):
            return np.sum(safe_log(p))

        def grad_J(theta, X, T, p):
            g = 0
            if l1:
                g = self.rho * (theta.T + u_z.T).reshape(-1)
            grad = np.array([np.dot(X.T, p[:, i] - T[:, i])
                             for i in range(self.K)]).reshape(-1)
            return grad + g

        def hess_J(theta, X, T, p):
            I = np.identity(self.K)
            H = np.zeros((self.K * self.K, self.n_features, self.n_features))

            for k in range(self.K):
                for m in range(self.K):
                    for n in range(self.n_samples):
                        if m == k:
                            temp = p[n, m] * (I[m, k] - p[n, k])
                            H[m + k * self.K] += temp * X[n].reshape(self.n_features, 1) * X[
                                n].reshape(1, self.n_features)  # 縦ベクトルx横ベクトル
            hess = np.zeros(self.n_features * self.K * self.n_features *
                            self.K).reshape(self.n_features * self.K, self.n_features * self.K)
            for i in range(self.K):
                hess[i * self.n_features:(i + 1) * self.n_features, i * self.n_features:(
                    i + 1) * self.n_features] = H[i + i * self.K]
            g = 0
            if l1:
                g = np.diag(
                    [self.rho for i in range(self.n_features * self.K)])
            return hess + g

        for iter_ in range(self.iter_max):
            p = calp(theta, X)
            hess = hess_J(theta, X, y, p)
            grad = grad_J(theta, X, y, p)
            theta_new = theta - \
                np.dot(np.linalg.inv(hess), grad).reshape(self.K, -1).T

            if iter_ != 0:
                diff = np.linalg.norm(theta_new - theta) / \
                    np.linalg.norm(theta)
                if diff < 0.01:
                    break
            theta = theta_new
        if self.verbose:
            print "iter: ", iter_,
        return theta.reshape(-1, self.K)

    def fit(self, X, y):
        self.classes_ = (np.unique(y))
        X_ = np.hstack((np.ones((X.shape[0], 1.)), X))  # inser intercept line
        y_ = np.array([[1 if i == j else 0 for i in y]
                       for j in self.classes_[:]]).T
        self.n_samples, self.n_features = X_.shape
        self.K = len(self.classes_)

        if self.penalty == "l1":
            self.fit_l1(X_, y_)
            return 0
        elif self.penalty == "l1_group":
            self.fit_l1_group(X_, y_)
            return 0
        elif self.penalty == "l1_group_l1":
            self.fit_l1_group_l1(X_, y_)
            return 0
        elif self.penalty == None:
            self.fit_normal(X_, y_)
            return 0
        else:
            raise ValueError("penalty is invalid")

    def fit_normal(self, X, y):
        theta = np.zeros((self.n_features, self.K))
        theta = self.cal_theta(theta, X, y).T

        if theta.shape[0] == 2:
            theta = theta[np.newaxis, 1]
        self.coef_ = theta[:, 1:]
        self.intercept_ = theta[:, 0]

    def fit_l1(self, X, y):
        theta = np.zeros((self.n_features, self.K))
        u = np.zeros((self.n_features, self.K))
        z = np.zeros((self.n_features, self.K)) + 1.

        for iter_a in range(self.iter_admm):
            z_ = z.copy()
            theta = self.cal_theta(theta, X, y, u_z=u - z, l1=True)

            # update z
            z[1:] = self.SoftMax(self.alpha / self.rho, (theta + u)[1:])
            z[0] = (theta + u)[0]
            u = u + theta - z

            diff = np.linalg.norm(z - z_) / np.linalg.norm(z_)
            self.curve.append(diff)
            if self.verbose:
                print "epoc: ", iter_a, "diff: ", diff, "\nz: ", (z.T[0])
                z_save = z.T
                if z_save.shape[0] == 2:
                    z_save = z_save[np.newaxis, 1]
                self.coef_ = z_save[:, 1:]
                self.intercept_ = z_save[:, 0]
                st.savepickle(".temp/%05d.pkl" % iter_a, self)
            if diff < self.error:
                break
            # self.plot(X[:,1:],y,z.T,"out/plt_%03d.png"%iter_a)

        z_save = z.T
        if z_save.shape[0] == 2:
            z_save = z_save[np.newaxis, 1]
        self.coef_ = z_save[:, 1:]
        self.intercept_ = z_save[:, 0]

    def fit_l1_group(self, X, y):

        theta = np.zeros((self.n_features, self.K))
        u = np.zeros((self.n_features, self.K))
        z = np.zeros((self.n_features, self.K)) + 1.

        for iter_a in range(self.iter_admm):
            z_ = z.copy()
            theta = self.cal_theta(theta, X, y, u_z=u - z, l1=True)

            # update z
            for i in range(len(z)):
                if i == 0:
                    z[0] = (theta + u)[0]
                    continue
                z[i] = self.GroupSoftMax(self.alpha / self.rho, (theta + u)[i])
            u = u + theta - z

            diff = np.linalg.norm(z - z_) / np.linalg.norm(z_)
            self.curve.append(diff)
            if self.verbose:
                print "epoc: ", iter_a, "diff: ", diff, "\nz: ", (z.T[0])
                z_save = z.T
                if z_save.shape[0] == 2:
                    z_save = z_save[np.newaxis, 1]
                self.coef_ = z_save[:, 1:]
                self.intercept_ = z_save[:, 0]
                st.savepickle(".temp/%05d.pkl" % iter_a, self)
            if diff < self.error:
                break
            # self.plot(X[:,1:],y,z.T,"out/plt_%03d.png"%iter_a)

        z = z.T
        if z.shape[0] == 2:
            z = z[np.newaxis, 1]
        self.coef_ = z[:, 1:]
        self.intercept_ = z[:, 0]

    def fit_l1_group_l1(self, X, y):

        theta = np.zeros((self.n_features, self.K))
        u = np.zeros((self.n_features, self.K))
        z = np.zeros((self.n_features, self.K)) + 1.

        for iter_a in range(self.iter_admm):
            z_ = z.copy()
            theta = self.cal_theta(theta, X, y, u_z=u - z, l1=True)

            # update z
            for i in range(len(z)):
                if i == 0:
                    z[0] = (theta + u)[0]
                    continue
                z[i] = self.GroupSoftMax(self.alpha / self.rho, (theta + u)[i])
            u = u + theta - z

            diff = np.linalg.norm(z - z_) / np.linalg.norm(z_)
            self.curve.append(diff)
            if self.verbose:
                print "epoc: ", iter_a, "diff: ", diff, "score: ", self.score(X, y), "\nz: ", (z.T[0])
                z_save = z.T
                if z_save.shape[0] == 2:
                    z_save = z_save[np.newaxis, 1]
                self.coef_ = z_save[:, 1:]
                self.intercept_ = z_save[:, 0]
                st.savepickle(".temp/%05d.pkl" % iter_a, self)
            if diff < self.error:
                break
            # self.plot(X[:,1:],y,z.T,"out/plt_%03d.png"%iter_a)

        z = z.T
        if z.shape[0] == 2:
            z = z[np.newaxis, 1]
        self.coef_ = z[:, 1:]
        self.intercept_ = z[:, 0]

    def plot(self, X, y, W_t, filename):
        def f(x1, W_t, c1, c2):
            a = - ((W_t[c1, 1] - W_t[c2, 1]) / (W_t[c1, 2] - W_t[c2, 2]))
            b = - ((W_t[c1, 0] - W_t[c2, 0]) / (W_t[c1, 2] - W_t[c2, 2]))
            return a * x1 + b

        def ff(W_t, c1, c2, color="green"):
            yy = (W_t[c1, 0] - W_t[c2, 0]) / (W_t[c1, 1] - W_t[c2, 1])
            plt.axvline(x=yy, color=str(color))

        plt.plot(X[:100, 0], X[:100, 1], "o", color="red")
        plt.plot(X[100:200, 0], X[100:200, 1], "o", color="blue")
        plt.plot(X[200:, 0], X[200:, 1], "o", color="green")

        # print W_t
        x1 = np.linspace(-20, 20, 1000)
        x2 = [f(x, W_t, 0, 1) for x in x1]
        if x2[0] == "inf":
            ff(W_t, 0, 1, color="red")
        else:
            plt.plot(x1, x2, 'r-')

        x1 = np.linspace(-20, 20, 1000)
        x2 = [f(x, W_t, 1, 2) for x in x1]
        if x2[0] == "inf":
            ff(W_t, 1, 2, color="blue")
        else:
            plt.plot(x1, x2, 'b-')

        x1 = np.linspace(-20, 20, 1000)
        x2 = [f(x, W_t, 2, 0) for x in x1]
        if x2[0] == "inf":
            ff(W_t, 2, 0, color="green")
        else:
            plt.plot(x1, x2, 'g-')

        # ff(W_t,1,2,color="blue")
        # ff(W_t,2,0,color="green")

        plt.xlim(np.min(X[:, 0]), np.max(X[:, 0]))
        plt.ylim(np.min(X[:, 1]), np.max(X[:, 1]))
        plt.savefig(filename)
        plt.close()

        # plt.savefig("output/a.pdf")
        # plt.close()
