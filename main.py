import pandas as pd

credits = pd.read_csv("data/credits.csv")
partition_size = int(credits.shape[0] / 4)
print(partition_size)
print("hola holas")