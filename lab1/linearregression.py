import numpy
import sklearn.preprocessing
from sklearn import datasets
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def f(x0, X, y):
    Xt = X.transpose()
    t = (y - Xt*x0)
    u = t.transpose()
    final = (np.matmul(t, u)) / 2425
    return final

if __name__ == '__main__':

    X, y = datasets.make_regression(n_features=1, noise=16, n_samples=2425, random_state=244825)

    poly = PolynomialFeatures(1)
    X = poly.fit_transform(X)

    print(len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.2, random_state=244825)

    Xt = np.transpose(X_train)

    w = np.matmul(np.linalg.inv(np.matmul(Xt, X_train)), (np.matmul(Xt, y_train)))
    w2 = np.linalg.inv(X_train.transpose().dot(X_train)).dot(X_train.transpose().dot(y_train))

    print("moj", w)
    print("moj2", w2)

    zzz = [1]
    #
    pp = optimize.minimize(f, zzz, args=(X_train, y_train), method='Powell')
    print(pp)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)

    # ttt = X_test * w2

    print("Model scikit", reg.coef_, reg.intercept_)
# print("---------")
# print(np.transpose(ttt))
# print(y_test)
