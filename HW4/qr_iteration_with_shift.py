# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 22:51:38 2021

@author: duola
"""

import numpy as np
from time import *
import matplotlib.pyplot as plt

def householder_reflection(A):
    (row,col)=np.shape(A)
    Q=np.eye(row)
    R=A
    for k in range(row-1):
        if np.linalg.norm(R[k+1:, k], ord=2)==0:
            continue
        else:
            x=R[k:, k]                  #取第k列第k行及其以下位置的向量
            e=np.zeros_like(x, dtype=float)
            e[0]=np.linalg.norm(x, ord=2)
            v=x+np.sign(x[0])*e
            v=v/np.linalg.norm(v, ord=2)
            H_k=np.eye(row)
            H_k[k:, k:]=H_k[k:, k:]-2*np.outer(v, v)            
            R=np.matmul(H_k, R)  # R=H_n-1...H_1A
            Q=np.matmul(Q, H_k)   # Q=(H_n-1...H_1)^T=H_1...H_n-1(因为Q是正交投影)
    return Q,R

def qr_iteration_with_shift(A, K, mu):
    (row,col)=np.shape(A)
    I=np.eye(row)
    
    error=[]
    
    for k in range(K):
        mu_k=mu
        Q,R=householder_reflection(A-mu_k*I)
        lastA=A
        A=np.matmul(R,Q)+mu_k*I
        error.append(np.linalg.norm(np.diagonal(A-lastA), ord=2))
    return A, error


B=np.zeros((10,10))         #生成B
for i in range(10):
    for j in range(10):
        if i+j==9:
            B[i][j]=1
            
B_k, error = qr_iteration_with_shift(B, 100, 2)
plt.title("error vs iteration")  
plt.xlabel("iterations")
plt.ylabel("error")
plt.plot(error, label="qr_iteration_with_shift")
plt.legend()
plt.show()