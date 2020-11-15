from sklearn import preprocessing
import numpy as np

x_train = np.array([[1., -1, 2],
                    [2, 0, 0],
                    [0, 1, -1]])

min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
print(x_train_minmax)

x_test = np.array([[-3., -1, 4]])  # 数据标准化
x_test_minmax = min_max_scaler.transform(x_test)
print(x_test_minmax)

# 缩放因子等属性
print(min_max_scaler.scale_)

print(min_max_scaler.min_)

print(min_max_scaler.data_max_)

print(min_max_scaler.data_min_)