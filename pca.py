from sklearn import datasets
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


iris = datasets.load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="FlowerType")

pca_iris = PCA(n_components=3).fit(iris.data)

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
iris_reduced = PCA(n_components=3).fit_transform(iris.data)

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    iris_reduced[:, 0],
    iris_reduced[:, 1],
    iris_reduced[:, 2],
    cmap=plt.cm.Paired,
    c=iris.target
)
for k in range(3):
    ax.scatter(
        iris_reduced[y == k, 0],
        iris_reduced[y == k, 1],
        iris_reduced[y == k, 2],
        label=iris.target_names[k]
    )

ax.set_title("First three P.C.")
ax.set_xlabel("P.C. 1")
ax.set_ylabel("P.C. 2")
ax.set_zlabel("P.C. 3")
plt.legend(numpoints=1)
plt.show()
