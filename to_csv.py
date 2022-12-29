import pandas as pd
import warnings   
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




sns.set_style('dark')

warnings.filterwarnings('ignore')

from scipy.io import loadmat  

data= loadmat("signal.mat")
transformed=data["signal"]



columns = []
index = range(6000)

for t in range(250):
    columns.append("signal"+str(t))

columns.append("label")

df = pd.DataFrame(index=index, columns = columns)





for j in range(3000):
    df["label"][j] = 0
    for i in range(250):
        yazi = "signal"+str(i)
        df[yazi][j] = np.array(data["signal"][0, i ,j])
    


for j in range(3000,6000):
    df["label"][j] = 1
    k = j-4000
    for i in range(250):
        yazi = "signal"+str(i)
        df[yazi][j] = np.array(data["signal"][1, i, k])


df.to_csv("train_data3bin.csv", encoding="utf-8", index=False)


print(df.head())

print(df)

print(df.info())

print(df.head())

print(df.groupby("label").count())