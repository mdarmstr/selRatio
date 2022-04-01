# This is a sample Python script.
import pandas as pd
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def selectratio():

    cottonPoly = pd.read_csv("coffee.csv", sep=",", header=0)
    X = cottonPoly.iloc[:, 1:].to_numpy()
    y = pd.get_dummies(cottonPoly.iloc[:, 0], drop_first=True).to_numpy(dtype="int32", copy=True).reshape(-1)

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.5, random_state=42)

    sz = np.shape(xTrain)

    selRatios = np.zeros((1, sz[1]))
    vipScrs = np.zeros((1, sz[1]))

    pls = PLSRegression(n_components=2)

    (x_scorest, y_scorest) = pls.fit_transform(xTrain, yTrain)
    (x_scoresT, y_scoresT) = pls.transform(xTest, yTest)

    x_loadings = pls.x_loadings_

    mdl = x_scorest @ x_loadings.T

    eR = xTrain - x_scorest @ x_loadings.T

    for vrbl in range(sz[1]):
        selRatios[0, vrbl] = np.linalg.norm(mdl[:, vrbl]) / np.linalg.norm(eR[:, vrbl])

    selRatios = np.nan_to_num(selRatios).reshape(-1)

    #VIP scores :)

    #b = pls.coef_
    xw = pls.x_weights_
    nmtr = 0
    dmtr = 0
    vipScrs = []

    b = np.linalg.pinv(x_scorest) @ yTrain
    xw = (xw / np.linalg.norm(xw, axis=0)) ** 2

    for vrbl in range(sz[1]):
        # for kk in range(np.size(x_scorest, 1)):
        nmtr = np.sum((b**2) @ x_scorest.T @ x_scorest @ xw[vrbl, :])
        dmtr = np.sum((b**2) @ x_scorest.T @ x_scorest)
        vipScrs.append(np.sqrt((sz[1] * nmtr) / dmtr))
        # dmtr = 0
        # nmtr = 0

    vipScrs.sort()
    print(vipScrs)
    plt.plot(vipScrs)
    plt.show()
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
    selInd = np.argsort(selRatios)[-10:]

    xTrainSel = xTrain[:, selInd]
    xTestSel = xTest[:, selInd]

    stdScaling2 = preprocessing.StandardScaler()
    xTrainSelScaled = stdScaling2.fit_transform(xTrainSel)
    xTestSelScaled = stdScaling2.transform(xTestSel)

    pca2 = PCA(n_components=2)
    SelScrs1 = pca2.fit_transform(xTrainSelScaled)
    SelScrs2 = pca2.transform(xTestSelScaled)

    axs[0, 1].scatter(SelScrs1[:, 0], noSelScrs1[:, 1], c=yTrain, alpha=0.35)
    axs[0, 1].scatter(SelScrs2[:, 0], noSelScrs2[:, 1], c=yTest, marker="*", alpha=0.35)

    axs[1, 0].scatter(x_scorest[:, 0], x_scorest[:, 1], c=yTrain, alpha=0.35)
    axs[1, 0].scatter(x_scoresT[:, 0], x_scoresT[:, 1], c=yTest, marker="*", alpha=0.35)


    #let's try it with the

    plt.show()


if __name__ == "__main__":
    selectratio()
