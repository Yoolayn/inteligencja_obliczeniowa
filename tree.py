import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("iris.csv")

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

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the decision tree classifier on the training set
clf.fit(train_inputs, train_classes)

# Calculate the accuracy of the classifier on the test set
accuracy = clf.score(test_inputs, test_classes)
print("Accuracy:", accuracy)

# Predict the classes for the test set
predicted_classes = clf.predict(test_inputs)

# Compute the confusion matrix
conf_matrix = confusion_matrix(test_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=train_inputs.columns,
    class_names=train_classes.unique(),
    filled=True,
)
plt.show()
