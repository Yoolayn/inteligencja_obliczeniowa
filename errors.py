import pandas as pd

names = {
    "setosa": "Setosa",
    "Versicolour": "Versicolor",
    "virginica": "Virginica",
}

if __name__ == "__main__":
    missing_values = ["-", "NA"]
    df = pd.read_csv("data/iris_with_errors.csv", na_values=missing_values)
    count = 0

    for row in df["variety"]:
        if row in names:
            df.loc[count, "variety"] = names[row]
        count += 1
    df["sepal.length"] = df["sepal.length"].mask(df["sepal.length"] < 0)
    df["sepal.width"] = df["sepal.width"].mask(df["sepal.width"] < 0)
    df["petal.length"] = df["petal.length"].mask(df["petal.length"] < 0)
    df["petal.width"] = df["petal.width"].mask(df["petal.width"] < 0)

    df["sepal.length"] = df["sepal.length"].mask(df["sepal.length"] > 15)
    df["sepal.width"] = df["sepal.width"].mask(df["sepal.width"] > 15)
    df["petal.length"] = df["petal.length"].mask(df["petal.length"] > 15)
    df["petal.width"] = df["petal.width"].mask(df["petal.width"] > 15)

    print(df.isnull().sum())

    median_sl = df["sepal.length"].median()
    median_sw = df["sepal.width"].median()
    median_pl = df["petal.length"].median()
    median_pw = df["petal.width"].median()

    df["sepal.length"] = df["sepal.length"].fillna(median_sl)
    df["sepal.width"] = df["sepal.width"].fillna(median_sw)
    df["petal.length"] = df["petal.length"].fillna(median_pl)
    df["petal.width"] = df["petal.width"].fillna(median_pw)

    print(df.isnull().sum())
    # sepal.length    2  sepal.length    2
    # sepal.width     1  sepal.width     3
    # petal.length    0  petal.length    0
    # petal.width     1  petal.width     1
    # variety         1  variety         1
    print(df.values)
