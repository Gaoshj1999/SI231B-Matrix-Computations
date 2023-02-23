# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:47:08 2021

@author: duola
"""

import numpy as np
A=np.asarray([[1,-1,0,0],[0,0,1,1]])
B=np.asarray([[1,-1,100000,100000],[0,0,1,1]])
A_cond=np.linalg.cond(A, p=2)
B_cond=np.linalg.cond(B)
print(A_cond)
print(B_cond)