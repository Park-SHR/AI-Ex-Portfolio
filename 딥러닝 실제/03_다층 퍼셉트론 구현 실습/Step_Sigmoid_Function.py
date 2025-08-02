# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:37:24 2025

@author: sims
"""
import numpy as np
import matplotlib.pylab as plt

# 계단 함수
def step_function(x):
    return np.array(x > 0, dtype=int)

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # ← 괄호 닫음!

# ReLU 함수
def relu(x):
    return np.maximum(0,x)

# x축 범위 설정
x = np.arange(-5.0, 5.0, 0.1)

# 각 함수 결과 계산
y1 = step_function(x)
y2 = sigmoid(x)
y3 = relu(x)

# 그래프 그리기
plt.plot(x, y1, linestyle='dashed', label='Step Function')  # 점선
plt.plot(x, y2, linestyle='solid', label='Sigmoid Function')  # 실선
plt.plot(x, y3, linestyle='dotted', label='ReLU Function')  # 실선
plt.ylim(-0.1, 1.1)
plt.title("Step vs Sigmoid vs ReLU Function")
plt.legend()
plt.show()