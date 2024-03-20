from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# a) Podział zbioru danych na testowy (30%) i treningowy (70%)
data = load_iris()
# data = load_diabetes()
# print(data)
X = data.data
Y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=388496
)

# b) Budowa modelu sieci z dwiema warstwami ukrytymi
model = MLPClassifier(
    hidden_layer_sizes=(6, 3),
    activation="relu",
    max_iter=500,
    learning_rate_init=0.1,
    random_state=388496
)

# c) Trenowanie modelu na zbiorze treningowym
model.fit(X_train, y_train)

# d) Ewaluacja na zbiorze testowym
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Dokładność modelu: {:.2f}%".format(accuracy * 100))
print("Macierz błędu:")
print(conf_matrix)

# e) Porównanie z poprzednimi klasyfikatorami

model2 = MLPClassifier(
    hidden_layer_sizes=(10, 8, 5),
    activation="tanh",
    max_iter=1500,
    learning_rate_init=1,
    random_state=388496
)

model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
conf_matrix2 = confusion_matrix(y_test, y_pred2)

print("Dokładność modelu: {:.2f}%".format(accuracy2 * 100))
print("Macierz błędu:")
print(conf_matrix2)
# f) Odpowiedź na pytanie
