# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 20:10:44 2021

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
x_axis = []
threshold = []
for i in range(rank_A):
    x_axis.append(i+1)
    threshold.append(150)
Squared_Z = Z*Z

plt.title("squared singular values & threshold=150")  
plt.xlabel("number")
plt.ylabel("value")
plt.plot(x_axis, Squared_Z, label="squared singular values")
plt.plot(x_axis, threshold, label="threshold")
plt.legend()
plt.show()