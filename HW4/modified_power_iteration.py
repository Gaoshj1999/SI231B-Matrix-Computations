# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 00:36:48 2021

@author: duola
"""

import numpy as np
from time import *
import matplotlib.pyplot as plt

def power_iteration(q,A,tol,K):
    row=np.shape(A)[0]              #返回A的shape    
    col=np.shape(A)[1]
    error=1
    Lambda=float("inf")             #默认初始最大特征值为无穷大
    k=0
    while error>tol and k<K:
        k=k+1
        z=np.matmul(A,q)
        lastq=q
        q=z/np.linalg.norm(z,ord=2)
        Lambda=np.matmul(np.transpose(q),np.matmul(A,q))[0][0] #取矩阵中的元素
        error=np.linalg.norm(q-lastq, ord = 2)
    return Lambda, k
    
def modified_power_iteration(q,A,tol,K):
    row=np.shape(A)[0]              #返回A的shape    
    col=np.shape(A)[1]
    error=1
    Lambda=float("inf")             #默认初始最大特征值为无穷大
    k=0
    V=A
    while error>tol and k<K:
        k=k+1
        U=np.matmul(V,V)
        lastV=V
        V=U/np.trace(U)
        q=np.matmul(V,q)
        Lambda=(np.matmul(np.transpose(q),np.matmul(A,q))/ \
                np.matmul(np.transpose(q),q))[0][0] #取矩阵中的元素
        error=np.linalg.norm(V-lastV, ord = 2)  
    return Lambda, k
    
def q0(A):                                       # 随机选取一个初始向量
    col=np.shape(A)[1]                           # 返回A的shape    
    q=np.random.rand(col).reshape(col,1)      
    q=q/np.linalg.norm(q,ord=2)     #||q||=1
    return q

B = np.array([[10,1,2,3,4],[1,9,-1,2,-3],[2,-1,7,3,-5],[3,2,3,12,-1],[4,-3,-5,-1,15]])

tl_for_power=[]
tl_for_modified=[]
totaltl_for_power=0
totaltl_for_modified=0

true_eigenvalue=np.linalg.eig(B)[0][0]

for i in range(10):
    q=q0(B)
    a,b=power_iteration(q,B,1e-7,100)
    c,d=modified_power_iteration(q,B,1e-7,100)
    totaltl_for_power=totaltl_for_power+abs(a-true_eigenvalue)
    totaltl_for_modified=totaltl_for_modified+abs(c-true_eigenvalue)
    tl_for_power.append(totaltl_for_power)
    tl_for_modified.append(totaltl_for_modified)

plt.title("cumulative error")  
plt.xlabel("iterations")
plt.ylabel("tol")
plt.plot(tl_for_power,label="power_method")
plt.plot(tl_for_modified,label="modified_power_method")
plt.legend()
plt.show()