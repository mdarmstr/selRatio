# This is a sample Python script.
from sklearn.datasets import fetch_covtype
from sklearn.cross_decomposition import PLSRegression

X, y = fetch_covtype(return_X_y=True)

