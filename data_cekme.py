import pandas as pd
from scipy.io import loadmat
import numpy as np
data= loadmat("transformed.mat")
transformed=data["transformed"]

print(len(data))
print(type(data))

print("POP")

print(type(data["transformed"]))


print(data["transformed"].shape)

#print(data["transformed"][1][:][99])


sinyal = data["transformed"][1,:,999]
print(sinyal)
print(type(sinyal))

print(len(sinyal))




columns = ["signal", "label", range(250)]
index = range(2000)
signal_array = range(250)

df = pd.DataFrame(index=index, columns = columns)
print(df)




for j in range(1000):
    df["label"][j] = 0
    df["signal"][j] = np.array(data["transformed"][0, : ,j])


for j in range(1000,2000):
    df["label"][j] = 1
    k = j-1000
    df["signal"][j] = np.array(data["transformed"][1, :, k])



print(df["signal"][1500])


print(type(df["signal"][1500]))


print(len(df["signal"][1500]))


