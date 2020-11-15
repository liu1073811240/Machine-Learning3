from sklearn import preprocessing
import numpy as np

""""
6.Normalizer()
norm：可以为l1、l2或max，默认为l2
若为l1时，样本各个特征值除以各个特征值的绝对值之和
若为l2时，样本各个特征值除以各个特征值的平方之和
若为max时，样本各个特征值除以样本中特征值最大的值
"""

x = np.array([[1, -1, 2.],
              [2., 0, 0],
              [0, 1, -1]])

scaler = preprocessing.Normalizer(norm="l2")
x_scale = scaler.fit_transform(x)  # 先拟合数据，再标准化
print(x_scale)
print(x_scale.mean(0), x_scale.std(0))


# 数值二值化
scaler = preprocessing.Binarizer(threshold=0)
x_scale = scaler.fit_transform(x)
print(x_scale)
'''
[[1. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
'''

