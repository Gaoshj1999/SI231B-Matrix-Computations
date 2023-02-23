# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:49:13 2021

@author: duola
"""
import numpy as np
from time import *
import matplotlib.pyplot as plt

def generate_dense_symmertric_with_eigenvalue_of_D():   #随机生成矩阵A

    T=np.random.rand(10,10)                             #随机生成一个矩阵
    Q,R=np.linalg.qr(T)                                 #根据T随机生成正交矩阵Q
    
    D=np.eye(10)                                        #生成对角矩阵D
    for i in range(10):
        D[i][i]=D[i][i]+9-i
        
    A=np.matmul(Q,np.matmul(D,np.transpose(Q)))         #A=QDQ^(T)可以保证矩阵A与D有相同特征值且为对称矩阵

    return A

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

def qr_iteration(A, K):
    error=[]
    D=np.eye(10)                                        
    for i in range(10):
        D[i][i]=D[i][i]+9-i
        
    Time=[]
    totaltime=0
    
    for k in range(K):
        begin_time = time()
        Q,R=householder_reflection(A)
        A=np.matmul(R,Q)
        error.append(np.linalg.norm(np.diagonal(A-D), ord=2))
        endtime = time()
        runtime = endtime-begin_time
        totaltime=totaltime+runtime
        Time.append(totaltime)
    return A, error, Time

def hessenberg_reduction(A):
    (row,col)=np.shape(A)
    for k in range(row-2):
         x=A[k+1:,k]
         e=np.zeros_like(x,dtype=float)
         e[0]=np.linalg.norm(x, ord=2)
         v=x+np.sign(x[0])*e
         v=v/np.linalg.norm(v, ord=2)
         A[k+1:,k:]=A[k+1:,k:]-2*np.outer(v,np.matmul(np.transpose(v),A[k+1:,k:]))
         A[:,k+1:]=A[:,k+1:]-2*np.outer(np.matmul(A[:,k+1:],v),np.transpose(v))
    return A

def hessenberg_qr_iteration(A, K):
    error=[]
    D=np.eye(10)                                        
    for i in range(10):
        D[i][i]=D[i][i]+9-i
    
    H=hessenberg_reduction(A)
    
    Time=[]
    totaltime=0

    for k in range(K):
        begin_time = time()
        Q,R=householder_reflection(H)
        H=np.matmul(R, Q)
        error.append(np.linalg.norm(np.diagonal(H-D), ord=2))
        endtime = time()
        runtime = endtime-begin_time
        totaltime=totaltime+runtime
        Time.append(totaltime)
    return H, error, Time

A = generate_dense_symmertric_with_eigenvalue_of_D()
A_qr, qr_error, qr_time = qr_iteration(A, 30)
A_hess, hess_error, hess_time = hessenberg_qr_iteration(A, 30)
plt.title("error vs iteration time")  
plt.xlabel("runtime")
plt.ylabel("error")
plt.plot(qr_time, qr_error, label="qr_iteration")
plt.plot(hess_time, hess_error, label="hessenberg_qr_iteration")
plt.legend()
plt.show()