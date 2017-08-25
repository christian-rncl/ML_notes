class LinearRegression():
    def __init__(self):
        self.ALPHA = .01
        self.EPSILON = 0
        self.ITER_LIMIT = 1500
        self.EPSILON = .00000001

    def add_bias_col(self, X):
        return np.column_stack((np.ones(X.shape[0]), X))

    def j(self, Theta, *args):
        X = args[0]
        y = args[1]
        [m, n] = X.shape
        loss = np.dot(X, Theta.reshape(n, 1)) - y
        return (1 / (2 * m)) * (sum(np.power(loss, 2)))

    def pred(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = self.add_bias_col(X)
        return np.dot(X, self.Theta.reshape(X.shape[1], 1))

    def check_error(self, h, y):
        m = len(h)
        loss = h.reshape(m, 1) - y.reshape(m, 1)
        return (1 / (2 * m)) * (sum(np.power(loss, 2)))

    def fit(self, X, y, debug):
        # if dataframe, convert to numpy array

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values.reshape(len(y.values), 1)
a
        X = self.add_bias_col(X)
        self.Theta = np.zeros((X.shape[1], 1))  # initialize Theta

        # minimize with BFGS using scipy optimize
        optim = minimize(self.j, self.Theta.reshape(X.shape[1], 1), method='BFGS', args=(X, y),
                         options={'disp': True if debug else False})
        self.Theta = optim['x']