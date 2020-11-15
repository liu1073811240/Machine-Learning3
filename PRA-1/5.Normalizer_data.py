from sklearn import preprocessing
import numpy as np

x = np.array([[1, -1, 2],
              [2, 0, 0.],
              [0, 1, -1]])

scaler = preprocessing.Normalizer(norm="l2")
x_scale = scaler.fit_transform(x)
print(x_scale)
print(x_scale.mean(0), x_scale.std(0))


# 数值二值化
scaler = preprocessing.Binarizer(threshold=0)
x_scale = scaler.fit_transform(x)
print(x_scale)


