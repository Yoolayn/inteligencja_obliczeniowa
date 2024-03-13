import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_csv("data/iris.csv")

# Split the dataset into training and testing sets (70%/30%)
train_set, test_set = train_test_split(df, test_size=0.3, random_state=288496)

# Display the training and testing sets
print("Training set:")
print(train_set)
print("\nTesting set:")
print(test_set)

# Split each set into input features and target class
train_inputs = train_set.drop("variety", axis=1)
train_classes = train_set["variety"]
test_inputs = test_set.drop("variety", axis=1)
test_classes = test_set["variety"]


def resultComp(tool, prediction, answers):
    accuracy = accuracy_score(prediction, answers)
    f1 = f1_score(prediction, answers, average="weighted")
    matrix = confusion_matrix(prediction, answers)
    scores = cross_val_score(tool, train_inputs, train_classes, cv=5)

    print(matrix)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Scores:", scores)
    print("--------------")


def knn(list):
    for k in list:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        print(knn)
        knn.fit(train_inputs, train_classes)

        output = knn.predict(test_inputs)

        print(f"{k}NN")
        resultComp(knn, output, test_classes)


knn([5, 5, 11])

bayes = GaussianNB()
bayes.fit(train_inputs, train_classes)
bayesOutput = bayes.predict(test_inputs)


print("Bayes")
resultComp(bayes, bayesOutput, test_classes)
