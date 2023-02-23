# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 20:51:58 2021

@author: duola
"""

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

data = scio.loadmat("data1.mat") #因为我是把data挂在同一个文件夹下了，所以如果要运行这个程序也需要如此
                                 #或者把A直接复制进来也行
A = data['data1']

U,Z,V_T = np.linalg.svd(A)
rank_A = Z.shape[0]
Squared_Z = Z*Z

x_axis = []
for i in range(rank_A-1):
    x_axis.append(i+1)
    
f_function = []
for i in range(rank_A-1):
    f_function.append(sum(Squared_Z[i+1:])/sum(Squared_Z))
print(f_function)

d=[1,2,3,4]
compression_rate=[]
for i in range(4):
    compression_rate.append((d[i]*(2016+10+1))/(2016*10))
print(compression_rate)

plt.title("f(d)")  
plt.xlabel("number")
plt.ylabel("value")
plt.plot(x_axis, f_function, label="f(d)")
plt.legend()
plt.show()
