# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:36:33 2021

@author: duola
"""

import numpy as np
A=np.asarray([[3,-4j],[5j,12]],dtype=complex)
U,Z,V_T=np.linalg.svd(A)
print(A)
print(U)
print(Z)
print(V_T)
print("////////////////////////////////////")
B=np.asarray([[3,0,0,4],[0,12,-5,0],[0,-4,3,0],[5,0,0,12]])
U_B,Z_B,V_B_T=np.linalg.svd(B)
print(B)
print(U_B)
print(Z_B)
print(V_B_T)

U_real=np.real(U)
U_imag=np.imag(U)
V_T_real=np.real(V_T)
Z_matrix=np.zeros((2,2))
Z_matrix[0][0]=13.88060922 
Z_matrix[1][1]=1.15268716
print("/////////////////////////")
print(U)
print("/////////////////////////")
print(np.real(U))
print("/////////////////////////")
print(np.imag(U))
print("/////////////////////////")
print(U_B)
print("/////////////////////////")
C=np.asarray([[3,-4j],[4j,3]])
a,b,c=np.linalg.svd(C)
print(a)
print(b)
print(c)
real=np.matmul(np.real(U),np.matmul(Z_matrix,np.real(V_T)))
imag=np.matmul(np.imag(U),np.matmul(Z_matrix,np.imag(V_T)))
print(real-imag)
test1=np.matmul(np.real(U),np.matmul(Z_matrix,np.imag(V_T)))
test2=np.matmul(np.imag(U),np.matmul(Z_matrix,np.real(V_T)))
print(test1+test2)
