# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:24:33 2025

@author: sims
"""

import numpy as np

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X,W1)+B1

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #활성화 함수 통과
Z1 = sigmoid(A1)

print('1번째 층')
print(A1)
print('활성화 통과')
print(Z1)

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

A2=np.dot(Z1,W2)+B2
print('2번째 층')
print(A2)

Z2 = sigmoid(A2)
print('활성화 통과')
print(Z2)

def identity_function(x):
    return x

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
Y=identity_function(A3)

print(Y)