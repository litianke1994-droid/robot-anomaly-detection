import pandas as pd

path = r"D:\10_AI_Industrial_Engineer\Project_01_ML\data\raw\voraus-ad-dataset-100hz.parquet"

columns = ["sample", "time", "anomaly", "motor_iq_1"]

df = pd.read_parquet(path, columns=columns)

print("数据的行数和列数：")
print(df.shape)

print("前十行数据：")
print(df.head(10))

df = df[df["sample"] == 755]   # 取出作业编号为755的行
print("作业编号为755的数据：")
print(df)  # 查看作业编号为755的数据