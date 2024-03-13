import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris.csv")

(train_set, test_set) = train_test_split(
    df.values,
    train_size=0.7,
    random_state=288496
)

print(df)
print(test_set)
print(test_set.shape[0])

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]


def classify_iris(sl, sw, pl, pw):
    if pw < 1:
        return "Setosa"
    if sl > 6 and pw >= 1.8:
        return "Virginica"
    return "Versicolor"


good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if (
        classify_iris(
            sl=test_inputs[i][0],
            sw=test_inputs[i][1],
            pl=test_inputs[i][2],
            pw=test_inputs[i][3],
        ) == test_classes[i]
    ):
        good_predictions = good_predictions + 1

print(good_predictions)
print(good_predictions / len * 100, "%")
