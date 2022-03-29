# This is a sample Python script.
import sklearn.decomposition
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import numpy as np

X, y = fetch_covtype(return_X_y=True)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.5, random_state=42)

sz = np.shape(xTrain)

selRatios = np.zeros((1, sz[1]))

pls = PLSRegression(n_components=5)

(x_scores, y_scores) = pls.fit_transform(xTrain, yTrain)

x_loadings = pls.x_loadings_

mdl = x_scores @ x_loadings.T

eR = xTrain - x_scores @ x_loadings.T

for vrbl in range(sz[1]):
    selRatios[0, vrbl] = np.linalg.norm(mdl[:, vrbl]) / np.linalg.norm(eR[:, vrbl])

# no selected variables
stdScaling = preprocessing.StandardScaler()
xTrainScaled = stdScaling.fit_transform(xTrain)
xTestScaled = stdScaling.transform(xTest)

pca1 = PCA(n_components=2)
noSelScrs1 = pca1.fit_transform(xTrainScaled)
noSelScrs2 = pca1.transform(xTestScaled)

fig, axs = plt.subplots(2, 2)

axs[0, 0].scatter(noSelScrs1[:, 0], noSelScrs1[:, 1], c=yTrain, alpha=0.35)
axs[0, 0].scatter(noSelScrs2[:, 0], noSelScrs2[:, 1], c=yTest, marker="*", alpha=0.35)
# axs[0, 0].title.set_text("PCA | Ext. Accuracy = %.2f" % pcaAcc)

# selected variables

selFeat = selRatios > 1

xTrainSel = xTrain[:, selFeat[0, :]]
xTestSel = xTest[:, selFeat[0, :]]

stdScaling = preprocessing.StandardScaler()
xTrainSelScaled = stdScaling.fit_transform(xTrainSel)
xTestSelScaled = stdScaling.transform(xTestSel)

pca2 = PCA(n_components=2)
SelScrs1 = pca1.fit_transform(xTrainSelScaled)
SelScrs2 = pca1.transform(xTestSelScaled)

plt.show()


