# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
#y = np.sin(x)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label = "sin")
plt.plot(x, y2, linestyle="--", label = "cos")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.title('sin & cos')
plt.axhline(0.5,color = 'black', linewidth = 0.5)
plt.legend()
plt.show()