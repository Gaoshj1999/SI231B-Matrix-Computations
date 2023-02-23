# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 01:09:25 2021

@author: duola
"""

import numpy as np
from time import *
import matplotlib.pyplot as plt
def generate(n):
    return np.random.rand(n,n),np.random.rand(n,1)

def swap(a,b):     #交换矩阵某两行函数
    for i in range(len(a)):
        t = a[i]
        a[i] = b[i]
        b[i] = t

def Size(A):
    return A.shape[0]

def check_and_change(A,i): 
    size = Size(A)       
    if A[i][i]==0:
        j=0
        for k in range(i,size):
            if A[k][i] != 0:
                j=k
        return j
    else:
        return i
   
def elim(A,B,i):  
    size = Size(A)
    for k in range(i+1,size):
        temp = A[k][i]/A[i][i]
        A[k] = A[k]-temp*A[i]
        B[k] = B[k]-temp*B[i]

def eliminv(A,B,i):    
    size = Size(A)
    for k in range(i-1,-1,-1):
        temp = A[k][i]/A[i][i]
        A[k] = A[k]-temp*A[i]
        B[k] = B[k]-temp*B[i]
        
def adjust(A,B):     #归一化
    size = Size(A)
    for i in range(size):
        temp=A[i][i]
        A[i][i]=1
        B[i]=B[i]/temp
        
def inv(A): 
    size = Size(A)
    Ainv = np.eye(size)                    
    for i in range(size-1):                     
        j=check_and_change(A,i)
        if i != j:
            swap(A[i],A[j])
            swap(Ainv[i],Ainv[j])              
        elim(A,Ainv,i) 
    for i in range(size-1,-1,-1):                     
        for k in range(i-1,-1,-1):               
            eliminv(A,Ainv,i)           
    adjust(A,Ainv)
    return Ainv
'''
#This is the part to generate the graph
Atime=[]
Asize=[]

for i in range(10):
    A,b=generate((i+1)*100)
    begin_time = time()
    Ainv=inv(A)
    x=Ainv@b
    end_time = time()
    run_time = end_time-begin_time
    Atime.append(run_time)
    Asize.append((i+1)*100)

plt.title("time for Inverse method")  
plt.xlabel("size")
plt.ylabel("time/s")
plt.plot(Asize,Atime)
plt.show()
'''

'''
#This is the part using for test
B,z=generate(2)
x1=np.linalg.solve(B,z)
Binv=inv(B)
x2=Binv@z
print(x1)
print(x2)
'''
