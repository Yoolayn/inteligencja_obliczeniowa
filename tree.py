import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("data/iris.csv")

train_set, test_set = train_test_split(df, test_size=0.3, random_state=288496)

print("Training set:")
print(train_set)
print("\nTesting set:")
print(test_set)

train_inputs = train_set.drop("variety", axis=1)
train_classes = train_set["variety"]
test_inputs = test_set.drop("variety", axis=1)
test_classes = test_set["variety"]

clf = DecisionTreeClassifier(random_state=42)

clf.fit(train_inputs, train_classes)

accuracy = clf.score(test_inputs, test_classes)
print("Accuracy:", accuracy)

predicted_classes = clf.predict(test_inputs)

conf_matrix = confusion_matrix(test_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=train_inputs.columns,
    class_names=train_classes.unique(),
    filled=True,
)
plt.show()
