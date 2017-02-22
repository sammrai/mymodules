import numpy as np
from mypool import MyPool
from sklearn import linear_model

class AdaptiveLasso(linear_model.LinearRegression):
    """
    Implementation of the adaptive lasso algorithm using ordinaly lasso algorithm.
    The optimization objective for Adaptive Lasso is::
        ||y - Xw||^2_2 + alpha * ||w||_1*v

    where v the weight vector is defined as following::

        v = 1/||w||^gamma_1

    The model of alpha is selected by cross-validation. Read reference to get more algorithm detail.

    Parameters
    ----------
    gamma : float
        The amount of penalization gamma. If None, adopt best score based on 10-fold cross varidation from gamma=[0.5, 1., 2.]
    alpha : float
        The amount of regulaization parameter. If None, adopt best score based on 10-fold cross varidation.
    alasso_iter_num : int
        The number of iteration
    verbose : boolean
        Print debug message

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    Examples
    --------
    >>> import reg
    >>> X=[[0,0], [1, 1], [2, 2]]
    >>> y=[0, 1, 2]
    >>> clf = AdaptiveLasso()
    >>> clf.fit(X,y)

    >>> print clf.coef_
    [ 0.999  0.   ]
    >>> print clf.predict([0.5,0.5])
    0.5005
    >>> print clf.score(X,y)
    0.999999

    References
    ----------
    "The Adaptive Lasso and Its Oracle Properties",Hui Zou, Scheninberg, http://pages.cs.wisc.edu/~shao/stat992/zou2006.pdf


    """

    def __init__(self, gamma=None, alpha=None, alasso_iter_num=10, verbose=False):
        self.gamma = gamma
        self.alpha = alpha
        self.alasso_iter_num = alasso_iter_num
        self.verbose = verbose
        # if self.verbose:
        # import warnings
        # warnings.simplefilter("ignore", DeprecationWarning)

    def fit_one(self, X, y):
        """Fit linear model with coordinate descent
        Fit is on grid of alpha and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Pass directly as float64, Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
        """
        from sklearn import linear_model
        X_ = np.copy(X)
        y_ = np.copy(y)
        clf = linear_model.LinearRegression()
        c_ = clf.fit(X_, y_).coef_

        A = 1
        for i in range(self.alasso_iter_num):
            w_ = np.power(np.abs(c_), self.gamma)
            X_ = X_ * w_
            A = A * w_
            clf = linear_model.Lasso(alpha=self.alpha)
            clf.fit(X_, y_)
            c_ = clf.coef_ * w_
        self.coef_ = clf.coef_ * A
        self.intercept_ = np.average(
            y_) - np.dot(np.mean(X, axis=0), self.coef_)

    def setalpha(self, X, y, epsilon=1e-17, K=100):
        lambdamax = np.max(np.abs(np.dot(X.T, y))) / len(y)
        return np.logspace(np.log10(epsilon * lambdamax), np.log10(lambdamax), num=K)

    def cvscore(self, X, y, gamma, alpha, fold_size):
        clf = AdaptiveLasso(gamma=gamma, alpha=alpha)
        from sklearn.cross_validation import KFold
        kf = KFold(len(y), fold_size, shuffle=True, random_state=0)
        sum = 0
        for one_kf in kf:
            train, test = one_kf
            X_train, X_test, y_train, y_test = X[
                train], X[test], y[train], y[test]
            clf.fit_one(X_train, y_train)
            sum += clf.score(X_test, y_test)
        return sum / len(kf)

    def fit(self, X, y, fold_size=5):
        if not self.gamma:
            gammas = [0.5, 1., 2.]
        else:
            gammas = np.expand_dims(self.gamma, axis=0)
        if not self.alpha:
            alphas = self.setalpha(X, y)
        else:
            alphas = np.expand_dims(self.alpha, axis=0)

        if len(gammas) == 1 and len(alphas) == 1:
            # if self.verbose: print "### MESSAGE ### : short route"
            self.gamma = gammas[0]
            self.alpha = alphas[0]
            self.fit_one(X, y)
            return 0
        # else:
            # if self.verbose: print "### MESSAGE ### : calling num of function
            # fit_one will be %d"%len(gammas)*len(alphas)

        alp_store = []
        for one_g in gammas:
            for one_a in alphas:
                alp_store.append(
                    [self.cvscore(X, y, one_g, one_a, fold_size=fold_size), one_g, one_a])
        _, gamma, alpha = alp_store[np.argmax(np.array(alp_store)[:, 0])]
        self.gamma = gamma
        self.alpha = alpha
        self.fit_one(X, y)

    def fit_(self, X, y, target, K=10, iter_=3):
        def process(one_a):
            clf_ = AdaptiveLasso(
                alpha=one_a, gamma=self.gamma, verbose=self.verbose)
            clf_.fit(X, y)
            return count(clf_.coef_)

        def getPinchindex(lis, target):
            if len(lis) < 2:
                print "#ERROR"
                raise
            for i in range(len(lis))[::-1]:
                if lis[i] >= target:
                    try:
                        return i, i + np.argmax(np.array(lis[i + 1:])) + 1
                    except:
                        return i - 1, i
            if lis[0] == target:
                return 0, 1
            raise ValueError(
                "#ERROR : target value is not exist in list. \nlis:%s target:%s" % (lis, target))

        def count(A, sikii=1e-12):
            cou = 1
            for i in A:
                if np.abs(i) < sikii:
                    cou += 1
            return len(A) - cou + 1

        pool = MyPool(K)

        for i in range(iter_):
            if i == 0:
                alphas = self.setalpha(X, y, K=K)
            else:
                alphas = np.logspace(
                    np.log10(alphas[pinch[0]]), np.log10(alphas[pinch[1]]), num=K)
            results = pool.map(process, alphas)
            pinch = getPinchindex(results, target)

            if self.verbose:
                print target, results, pinch
            if results[pinch[0]] == target:
                ind = pinch[0]
                break
            if target in results:
                ind = -results[::-1].index(target) + len(results) - 1
                break
        else:
            ind = pinch[0]

        self.alpha = alphas[ind]
        self.fit(X, y)


