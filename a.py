
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pywt


df = pd.read_csv("data_rand.csv")

print(df.shape)

data = df
#p1rint(data[0:1])


da = data[1996:1997]
#print(da.shape)
da2 = data[1805:1806]

X = da.drop('label', axis=1).copy()
X2 = da2.drop('label', axis=1).copy()



for family in pywt.families():
  print(family, ":", pywt.wavelist(family))

w=pywt.Wavelet("haar")

print(w)

print(type(X))

# X ve Y aynı şey yapıyoruz farklı yöntemlerle
X = X.to_numpy() 
X2 = X2.to_numpy() 
Y = np.array(X)

print(type(X))
print(type(Y))

X=X.flatten()
X2=X2.flatten()


cA, cD= pywt.dwt(X,"haar")
cA2, cD2= pywt.dwt(X2,"haar")


# print(cA)

# print(cD)


scales=np.arange(1,51)

coef,freqs=pywt.cwt(X,scales,"morl")

coef2,freqs2=pywt.cwt(X2,scales,"morl")


print(coef.shape)


print(coef)
plt.subplot(2,1, 1)
plt.imshow(coef)


plt.subplot(2, 1, 2)
plt.imshow(coef2)
plt.show()



print(coef.shape)
print(type(coef))





coeflistesi=[]
for i in range(len(data[:])):
    da = data[i:i+1]
    X = da.drop('label', axis=1).copy()
    X = X.to_numpy() 
    X=X.flatten()
    scales=np.arange(1,51)
    coef,freqs=pywt.cwt(X,scales,"morl")
    coeflistesi.append(coef)

print(len(coeflistesi[1][2]))

plt.imshow(coeflistesi[0])
plt.show()


