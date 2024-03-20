from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# b) Konwersja etykiet na liczby
# Wartości etykiet dla irysów to 0, 1, 2 odpowiadające kolejno
# "setosa", "versicolor", "virginica"

# c) Opcjonalne skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# d) Konstrukcja i trenowanie modelu MLP
# Topologia sieci: 4 neuronowa warstwa wejściowa, 2 neurony w jednej
# warstwie ukrytej, 3 klasy irysów w warstwie wyjściowej
mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# e) Ewaluacja modelu na zbiorze testowym
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu MLP na zbiorze testowym:", accuracy)

# f) Sprawdzenie modelu z trzema neuronami w warstwie ukrytej
mlp_3_neurons = MLPClassifier(
    hidden_layer_sizes=(3,),
    max_iter=1000,
    random_state=42
)
mlp_3_neurons.fit(X_train_scaled, y_train)
y_pred_3_neurons = mlp_3_neurons.predict(X_test_scaled)
accuracy_3_neurons = accuracy_score(y_test, y_pred_3_neurons)
print(
    "Dokładność modelu MLP z 3 neuronami w warstwie ukrytej:",
    accuracy_3_neurons
)

# g) Sprawdzenie modelu z dwiema warstwami po 3 neurony każda
mlp_2_layers = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    max_iter=1000,
    random_state=42
)
mlp_2_layers.fit(X_train_scaled, y_train)
y_pred_2_layers = mlp_2_layers.predict(X_test_scaled)
accuracy_2_layers = accuracy_score(y_test, y_pred_2_layers)
print(
    "Dokładność modelu MLP z dwiema warstwami po 3 neuronach każda:",
    accuracy_2_layers
)
