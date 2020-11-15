from sklearn import preprocessing
import numpy as np

"""
5.RobustScaler(),robust_scale()
    机器学习估计，通常是通过去除平均值来实现的然后缩放到单位方差。然而，异常值往往会影响
样本均值/方差。在这种情况下，中位数和四分位范围(RobustScaler)通常会得到更好的结果。
该缩放器删除中位数，并根据百分位数范围（默认值为IQR：四分位间距）缩放数据。IQR是第1个四
分位数（25%）和第3个四分位数（75%）之间的范围。
    适用：存在离群点数据，
"""

x = np.array([[1., -1, 2],
              [2., 1000, 0],
              [0, 1, -1]])

scaler = preprocessing.RobustScaler()
x_scale = scaler.fit_transform(x)  # 将噪声数据压缩到很小的一个范围

print(x_scale)
print(x_scale.mean(0), x_scale.std(0))
# [0.         0.66400266 0.22222222] [0.81649658 0.94186859 0.83147942]




