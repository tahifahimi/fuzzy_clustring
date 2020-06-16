import pandas as pd

if __name__=="__main__":
    # read file and initate the values
    df_full = pd.read_csv("sample2.csv")
    columns = list(df_full.columns)
    features = columns[:len(columns)]
    # print(features)
    class_labels = list(df_full[columns[-1]])
    df = df_full[features]
    print("the first coordinate", list(df.iloc[0]))
    # Number of Attributes
    num_attr = len(df.columns)
    print(num_attr)

    # Number of Clusters
    k = 4

    # Maximum number of iterations
    MAX_ITER = 10

    # Number of data points
    n = len(df)
    print("number of data points : ", n)
    # Fuzzy parameter
    m = 2.00
