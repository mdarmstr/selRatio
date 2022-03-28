# This is a sample Python script.
from sklearn.datasets import fetch_covtype
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split


import numpy as np

X, y = fetch_covtype(return_X_y=True)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.5, random_state=42)

sz = np.shape(xTrain)

selRatios = np.zeros((1, sz[1]))

pls = PLSRegression(n_components=5)

(x_scores, y_scores) = pls.fit_transform(xTrain, yTrain)

x_loadings = pls.x_loadings_

mdl = x_scores @ x_loadings.T

eR = X - x_scores @ x_loadings.T

for vrbl in range(sz[1]):
    selRatios[0, vrbl] = np.linalg.norm(mdl[:, vrbl]) / np.linalg.norm(eR[:, vrbl])
dfsd


wait = input("Press ENTER to continue")


