# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 16:21:07 2021

@author: duola
"""

import numpy as np
A=np.array([[0.986,0.579],[0.409,0.237]])

b=np.array([[0.235],[0.107]])

Ainv=np.linalg.inv(A)

A_norm=np.linalg.norm(A, ord = np.inf)
Ainv_norm=np.linalg.norm(Ainv, ord = np.inf)

(x1,x2)=np.linalg.solve(A,b)
x=np.array([[x1[0]],[x2[0]]])
x_norm=np.linalg.norm(x, ord=np.inf)
deltaA=np.array([[2,6],[4,8]])

u=[1e-1,1e-2,1e-4,1e-6,1e-8,1e-10]
deltax1=[]
deltax2=[]
upbound=[]
lowbound=[]
longterm=[]
print("A:", A, "A norm:",A_norm, "Ainv:",Ainv,"Ainv norm:",Ainv_norm, "x:",(x1,x2))
for i in range(len(u)):
    tempu=u[i]
    tempdeltaA=tempu*deltaA
    tempA=A+tempdeltaA
    (tempx1,tempx2)=np.linalg.solve(tempA,b)
    tempx=np.array([[tempx1[0]],[tempx2[0]]])
    deltax=x-tempx
    deltax_norm=np.linalg.norm(deltax, ord=np.inf)
    lowbound.append(deltax_norm/x_norm)
    deltax1.append(abs(x1[0]-tempx1[0]))
    deltax2.append(abs(x2[0]-tempx2[0]))
    tempdeltaA_norm=np.linalg.norm(tempdeltaA, ord=np.inf)
    templongterm=(Ainv_norm*tempdeltaA_norm)/(1-Ainv_norm*tempdeltaA_norm)
    longterm.append(templongterm)
    
print("//////////////")  
print(deltax1)
print("//////////////")  
print(deltax2)
print("//////////////")  
print(lowbound)
print("//////////////")  
print(longterm)