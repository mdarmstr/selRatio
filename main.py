# This is a script that performs a basic variable selection using Selectivity Ratios and VIP scores. The top 10 high-
# scoring variables are used for this demonstration

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA

from selrpy import selrpy
from vipy import vipy

import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Let's start with importing, splitting and scaling the data
    coffee = pd.read_csv("coffee.csv", sep=",", header=0)
    X = coffee.iloc[:, 1:].to_numpy()
    y = pd.get_dummies(coffee.iloc[:, 0], drop_first=True).to_numpy(dtype="int32", copy=True).reshape(-1)

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=42)

    stdscl = preprocessing.StandardScaler()
    xtrains = stdscl.fit_transform(xtrain)
    xtests = stdscl.transform(xtest)


    pca1 = PCA(n_components=2)
    tscrs1 = pca1.fit_transform(xtrains)
    tscrs2 = pca1.transform(xtests)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].scatter(tscrs1[:, 0], tscrs1[:, 1], c=ytrain, cmap="Accent", alpha=0.50)
    axs[0, 0].scatter(tscrs2[:, 0], tscrs2[:, 1], c=ytest, marker="*", cmap="Accent", alpha=0.50)
    axs[0, 0].title.set_text("PCA, all features")

    # Now let's do the feature selection with the selectivity ratio

    selr = selrpy(xtrains, ytrain, ncomponents=2)
    sel10idx = np.argsort(selr)[-10:]

    pca2 = PCA(n_components=2)
    tscrsSel1 = pca2.fit_transform(xtrains[:, sel10idx])
    tscrsSel2 = pca2.transform(xtests[:, sel10idx])

    axs[0, 1].scatter(tscrsSel1[:, 0], tscrsSel2[:, 1], c=ytrain, cmap="Accent", alpha=0.50)
    axs[0, 1].scatter(tscrsSel2[:, 0], tscrsSel2[:, 1], c=ytest, marker="*", cmap="Accent", alpha=0.50)
    axs[0, 1].title.set_text("PCA, SelRatio -10:")

    # Now let's select the top variables from VIP scores
    vipr = vipy(xtrains, ytrain, ncomponents=2)
    vip10idx = np.argsort(selr)[-10:]

    pca2 = PCA(n_components=2)
    tscrsvip1 = pca2.fit_transform(xtrains[:, vip10idx])
    tscrsvip2 = pca2.transform(xtests[:, vip10idx])

    axs[1, 0].scatter(tscrsvip1[:, 0], tscrsvip1[:, 1], c=ytrain, cmap="Accent", alpha=0.50)
    axs[1, 0].scatter(tscrsvip2[:, 0], tscrsvip2[:, 1], c=ytest, marker="*", cmap="Accent", alpha=0.50)
    axs[1, 0].title.set_text("PCA, VIP -10:")
    axs[1, 0].legend()

    axs[1, 1].axis("off")

    plt.show()



