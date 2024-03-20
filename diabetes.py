from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

df = pd.read_csv("data/diabetes.csv")

df["class"] = df["class"].map({"tested_negative": 0, "tested_positive": 1})

x = df.drop(columns=["class"])
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=3540
)

# b) Budowa modelu sieci z dwiema warstwami ukrytymi
model = MLPClassifier(
    hidden_layer_sizes=(6, 3),
    activation="relu",
    max_iter=500,
    learning_rate_init=0.1,
    random_state=27715
)

# c) Trenowanie modelu na zbiorze treningowym
model.fit(x_train, y_train)

# d) Ewaluacja na zbiorze testowym
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Dokładność modelu: {accuracy * 100:.2f}%")
print("Macierz błędu:")
print(conf_matrix)

# e) Porównanie z poprzednimi klasyfikatorami
model2 = MLPClassifier(
    hidden_layer_sizes=(10, 8, 5),
    activation="tanh",
    max_iter=1500,
    learning_rate_init=1,
    random_state=288501
)

model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test)
accuracy2 = accuracy_score(y_test, y_pred2)
conf_matrix2 = confusion_matrix(y_test, y_pred2)

print(f"Dokładność modelu: {accuracy2 * 100:.2f}%")
print("Macierz błędu:")
print(conf_matrix2)
# f) Odpowiedź na pytanie
# zależne od wymagań
