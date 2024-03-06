import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler

iris = datasets.load_iris()

X = iris.data[:, :2]

min_max_scaler = MinMaxScaler()
z_score_scaler = StandardScaler()

X_min_max = min_max_scaler.fit_transform(X)
X_z_score = z_score_scaler.fit_transform(X)

datasets = [
    (X, "Original Data"),
    (X_min_max, "Min-Max Normalized Data"),
    (X_z_score, "Z-Score Scaled Data"),
]

for data, title in datasets:
    print(f"{title}:")
    print(f"Min: {data.min(axis=0)}")
    print(f"Max: {data.max(axis=0)}")
    print(f"Mean: {data.mean(axis=0)}")
    print(f"Standard Deviation: {data.std(axis=0)}")
    print()

cmap = plt.get_cmap("viridis")

plt.figure(figsize=(18, 6))
for i, (data, title) in enumerate(datasets):
    plt.subplot(1, 3, i + 1)
    scatter = plt.scatter(
        data[:, 0],
        data[:, 1],
        c=iris.target,
        cmap=cmap
    )
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title(title)
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=list(iris.target_names)
    )

plt.tight_layout()
plt.show()
