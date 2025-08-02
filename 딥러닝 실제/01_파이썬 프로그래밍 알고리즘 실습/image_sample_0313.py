# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 21:19:17 2025

@author: sims
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread 

img = imread(r"D:\충북대\충북대학교 산업인공지능학과 석사과정\04_딥러닝실제_수업\2주차 수업내용\02_강의내용\photo_sample.jpg")

plt.imshow(img)
plt.show()