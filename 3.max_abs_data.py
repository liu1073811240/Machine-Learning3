from sklearn import preprocessing
import numpy as np

""" 4.MaxAbsScaler，最大绝对值,它不移动中心的数据，
这样不会破坏任何稀疏性。
x/np.abs(np.max(x))"""

x = np.array([[1., -1, 2],
              [2., 0, 0],
              [0, 1, 1]])

scaler = preprocessing.MaxAbsScaler()
x_scale = scaler.fit_transform(x)
print(x_scale)

out = x / np.abs(np.max(x, 0))
print(out)


