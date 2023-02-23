# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:00:09 2021

@author: duola
"""

import numpy as np
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
    #print("lambda:", Lambda)
    #print(k)
    return Lambda,k

def inverse_iteration(q,A,mu,tol,K):
    row=np.shape(A)[0]              #返回A的shape    
    col=np.shape(A)[1]
    error=1
    Lambda=float("inf")             #默认初始最大特征值为无穷大
    I=np.eye(col)   
    k=0
    while error>tol and k<K:
        k=k+1
        z=np.matmul(np.linalg.inv(A-mu*I),q)
        lastq=q
        q=z/np.linalg.norm(z,ord=2)
        Lambda=np.matmul(np.transpose(q),np.matmul(A,q))[0][0]
        error=np.linalg.norm(q-lastq, ord = 2)
    #print("lambda:", Lambda)
    #print(k)
    return Lambda,k

def q0(A):                                       # 随机选取一个初始向量
    col=np.shape(A)[1]                           # 返回A的shape    
    q=np.random.rand(col).reshape(col,1)      
    q=q/np.linalg.norm(q,ord=2)     #||q||=1
    return q

A = np.array([[0,100],[1,0]]) # 数组

q=q0(A)

lambda_for_power=[]
lambda_for_inverse=[]

for i in range(100):
    a,b=power_iteration(q,A,1e-10,i+1)
    c,d=inverse_iteration(q,A,4,1e-10,i+1)
    lambda_for_power.append(a)
    lambda_for_inverse.append(c)
    
plt.title("biggest eigenvalue under different iteration k")  
plt.xlabel("iterations")
plt.ylabel("eigenvalue")
plt.plot(lambda_for_inverse,label="inverse_method")
plt.plot(lambda_for_power,label="power_method")
plt.legend()
plt.show()
