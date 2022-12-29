import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [8,8]
plt.rcParams.update({"font.size": 10})

dx = 0.01
L= 2*np.pi
x = np.arange(0, L +dx, dx)
n = len(x)

nquart = int(np.floor(n/4))
f = np.zeros_like(x)
f[nquart: 3*nquart] = 1
A0 = np.sum(f * np.ones_like(x)) * dx * 2 / L
fFS = A0/2 * np.ones_like(f)

for k in range(1,101):
    Ak = np.sum(f * np.cos(2*np.pi*k*x/L)) * dx * 2 / L
    Bk = np.sum(f * np.sin(2*np.pi*k*x/L)) * dx * 2 / L
    fFS = fFS + Ak*np.cos(2*k*np.pi*x/L) + Bk*np.sin(2*k*np.pi*x/L)

plt.plot(x,f,color="k")
plt.plot(x,fFS,"-",color="r")
plt.show()