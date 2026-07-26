import cv2
import numpy as np

data = np.load('../03_data/stereo/case1/stereo.npy', allow_pickle=True).item()
E = data['E']
R1, R2, T = cv2.decomposeEssentialMat(E)

print('Rotation 1:')
print(R1)
print('Rotation 2:')
print(R2)
print('Translation:')
print(T)