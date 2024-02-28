import pandas as pd


if __name__ == "__main__":
    error = pd.read_csv("data/iris_with_errors.csv")
    help(error)
    # missing = error.isnull().sum()
    # print(missing)
    # for row in error["sepal.length"]:

