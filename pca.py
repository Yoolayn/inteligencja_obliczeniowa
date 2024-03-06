import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

num_features = X.shape[1]

for i in range(1, num_features):
    pca = PCA(n_components=i)
    X_pca = pca.fit_transform(X)
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"Liczba komponentów: {i}, wariancja: {explained_variance_ratio*100}%")
    if explained_variance_ratio >= 0.95:
        break

print(f"Optymalna liczba komponentów: {i}")

cmap = plt.colormaps.get_cmap("viridis")

if i == 2:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap=cmap)
    plt.title("PCA on Iris Dataset (2 components)")
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=list(iris.target_names)
    )
    plt.show()

elif i == 3:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=iris.target, cmap=cmap
    )
    ax.set_title("PCA on Iris Dataset (3 components)")
    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=list(iris.target_names),
        loc="upper right",
        title="Species",
    )
    ax.add_artist(legend)
    plt.show()
