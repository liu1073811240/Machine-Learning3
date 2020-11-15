from sklearn import preprocessing
import numpy as np

"3.preprocessing.MinMaxScaler(X),将属性缩放到一个指定的最大和最小值（通常是0-1）之间"

"""使用这种方法的目的包括：
1、对于方差非常小的属性可以增强其稳定性。
2、维持稀疏矩阵中为0的条目。
X_std=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
X_scaled=X_std/(max-min)+min
"""

x_train = np.array([[1., -1, 2],
                    [2, 0, 0],
                    [0, 1, -1]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(x_train)  # 先拟合数据，再转换数据
print(X_train_minmax)

x_test = np.array([[-3., -1, 4]])  # 数据标准化
x_test_minmax = min_max_scaler.transform(x_test)
print(x_test_minmax)  # [[-1.5         0.          1.66666667]]


# 缩放因子等属性
print(min_max_scaler.scale_)  # [0.5        0.5        0.33333333]

print(min_max_scaler.min_)  # [0.         0.5        0.33333333]

print(min_max_scaler.data_max_)  # [2. 1. 2.]

print(min_max_scaler.data_min_)  # [ 0. -1. -1.]