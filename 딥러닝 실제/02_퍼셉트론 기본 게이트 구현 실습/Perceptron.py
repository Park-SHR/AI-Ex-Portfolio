# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import numpy as np

# def AND(x1,x2):
#     x= np.array([x1,x2])
#     w = np.array([0.5, 0.5])
#     b = -0.7
#     tmp = np.sum(w*x) + b
    
#     if tmp < 0:
#         return 0
#     else:
#         return 1
    
# def NAND(x1, x2):
#     x= np.array([x1,x2])
#     w = np.array([-0.5, -0.5])
#     b = 0.7
#     tmp = np.sum(w*x) + b
    
#     if tmp <= 0:
#         return 0
#     else:
#         return 1
    
# def OR(x1, x2):
#     x= np.array([x1,x2])
#     w = np.array([0.5, 0.5])
#     b = -0.2
#     tmp = np.sum(w*x) + b
    
#     if tmp <= 0:
#         return 0
#     else:
#         return 1
        
# def XOR(x1, x2):
#     s1 = NAND(x1, x2)
#     s2 = OR(x1,x2)
#     y = AND(s1,s2)
#     return y


# NAND 기반 논리 게이트 정의
# def NAND(a: int, b: int) -> int:
#     return 0 if a == 1 and b == 1 else 1

# # 기본 게이트를 NAND만으로 정의
# def NOT(a: int) -> int:
#     return NAND(a, a)

# def AND(a: int, b: int) -> int:
#     return NOT(NAND(a, b))

# def OR(a: int, b: int) -> int:
#     return NAND(NOT(a), NOT(b))

# def XOR(a: int, b: int) -> int:
#     return AND(NAND(a, b), OR(a, b))

#ChatGPT사용 게이트 구성
# 계단 함수 정의
# def step(x):
#     return 1 if x > 0 else 0

# # AND 게이트 구현
# def AND(x1, x2):
#     w1, w2, b = 0.5, 0.5, -0.7  # 가중치와 편향
#     output = w1 * x1 + w2 * x2 + b
#     return step(output)

# # OR 게이트 구현
# def OR(x1, x2):
#     w1, w2, b = 0.5, 0.5, -0.2
#     output = w1 * x1 + w2 * x2 + b
#     return step(output)

# # NAND 게이트 구현
# def NAND(x1, x2):
#     w1, w2, b = -0.5, -0.5, 0.7
#     output = w1 * x1 + w2 * x2 + b
#     return step(output)

# # XOR 게이트 구현 (NAND, OR, AND 조합)
# def XOR(x1, x2):
#     s1 = NAND(x1, x2)
#     s2 = OR(x1, x2)
#     y = AND(s1, s2)
#     return y

# # 표 출력
# def print_gate_table():
#     print(f"{'x1':^3} {'x2':^3} | {'AND':^5} {'OR':^5} {'NAND':^6} {'XOR':^5}")
#     print("-" * 30)
#     for x1 in [0, 1]:
#         for x2 in [0, 1]:
#             and_out = AND(x1, x2)
#             or_out = OR(x1, x2)
#             nand_out = NAND(x1, x2)
#             xor_out = XOR(x1, x2)
#             print(f"{x1:^3} {x2:^3} | {and_out:^5} {or_out:^5} {nand_out:^6} {xor_out:^5}")

# # 실행
# if __name__ == "__main__":
#     print_gate_table()

#다중 퍼셉트론을 통한 XOR 구현
import numpy as np

# 계단 함수 (다차원 대응 가능)
def step(x):
    return np.where(x > 0, 1, 0)

# 은닉층 가중치와 편향
W_hidden = np.array([
    [-1, -1],  # NAND
    [1, 1]     # OR
])
b_hidden = np.array([1.5, -0.5])

# 출력층
W_output = np.array([1, 1])
b_output = -1.5

# XOR 퍼셉트론 구현
def xor_perceptron(x):
    x = np.array(x)
    h = step(np.dot(W_hidden, x) + b_hidden)
    y = step(np.dot(W_output, h) + b_output)
    return y.item()  # 스칼라로 반환

# 테스트
print("XOR 퍼셉트론 결과:")
print("x1 x2 | y")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        y = xor_perceptron([x1, x2])
        print(f"{x1}  {x2}  | {y}")