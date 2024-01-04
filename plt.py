# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt



length = 500
std = 0.5
x = np.random.normal(0.0, std, length)
y = np.random.normal(0.0, std, length)

r   = np.sqrt(x**2+y**2)
agl = np.arctan(y/x)

for k in range(length):
    if (x[k] < 0 and y[k] > 0) or (x[k] < 0 and y[k] < 0):
        agl[k] += np.pi
    elif (x[k] > 0 and y[k] < 0):
        agl[k] += 2*np.pi
    if x[k] == 0:
        agl[k] = np.pi/2 if y[k] > 0 else 3*np.pi/2
    elif y[k] == 0:
        agl[k] = 0 if x[k] > 0 else np.pi

# 振幅の値でアルファ値を変更
cv     = np.sqrt((agl-np.min(agl)) / (np.max(agl)-np.min(agl)+1e-12))
alphas = (r-np.min(r)) / (np.max(r)-np.min(r)+1e-12)
alphas **= 1.5
alphas[alphas<0.01] = 0.01
    
cmap = plt.cm.jet  
colors = [None for k in range(length)]
for k in range(length):
    c_r, c_g, c_b, _ = cmap(cv[k])
    colors[k] = (c_r, c_g, c_b, alphas[k])

plt.clf()
plt.scatter(x, y, c=colors, s=50, linewidths=1)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.grid()
plt.show()