import numpy as np
from scipy import optimize
from sklearn import linear_model

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


class LogisticRegression(linear_model.LogisticRegression):
    """
    Multiclass Logistic Regression inprementation using Newton-Raphson algorithm that includes regularlization term R(Ab).
    
    The optimization objective for Adaptive Lasso is::
    
        J(w) = -sum( ln(p(x|w)) + R(Ab) )

    where w is the coefficient we want, A is the Tikhonov matrix and p(x|w) is posterior probability defined as follwing using sigmoid function g(z)::

        p(x|w) = g(w^Tx)
        g(z) = 1/(1+exp(-z))
    When we asuume the regularization term l1, l1_group are described as follws::

        R(Ab) = alpha * ||w||_1
        R(Ab) = alpha * sum||b_i||_1

    where b_i is the group of each feature and this minimization will shrink for each feature group. 
    When remove regularization term, the optimization problem considered to be same. 

    Parameters
    ----------
    iter_admm : int
        The itelation number in ADMM convergence. This is varid when the penalty is specified.
    iter_max : int
        The itelation number in Newton-Raphson convergence.
    alpha : float
        The regularization parameter. As alpha increases parameter will be shrunk, 
        and some elements are shrunk to exact $0$ when alpha is sufficiently large.
    rho : float
        Parameters of convergence acceleration.
    penalty : [None|"l1"|"li_group"]
        The penalty term 
    tol : float
        Tolerance for termination.
    verbose : boolean
        Print debug message

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)
    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.
    curve : array, shape(iteration num)
        converge curve list.
    
    Examples
    --------

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X,y=iris.data,iris.target
    >>> import reg
    >>> clf=reg.LogisticRegression(penalty="l1_group")
    >>> clf.fit(X,y)

    >>> clf.coef_
    [[-0.          0.         -2.99227635 -0.01976993]
    [ 0.         -0.         -0.24579579 -0.03525203]
    [-0.         -0.          3.2343256   0.05500017]]
    >>> clf.score(X,y)
    0.953333333333

    References
    ----------------------------
    http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/logit.pdf
    http://statweb.stanford.edu/~tibs/ElemStatLearn/
    http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    """
    def __init__(self, iter_admm=300, iter_max=20, alpha=5., rho=1., penalty=None, tol=1e-4, verbose=False):
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
        self.tol = tol

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
                                n].reshape(1, self.n_features)
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
            if diff < self.tol:
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
            if diff < self.tol:
                break
            # self.plot(X[:,1:],y,z.T,"out/plt_%03d.png"%iter_a)

        z = z.T
        if z.shape[0] == 2:
            z = z[np.newaxis, 1]
        self.coef_ = z[:, 1:]
        self.intercept_ = z[:, 0]
