from sklearn import preprocessing
import numpy as np

''' 1.preprocessing.scale(X), 可以直接将给定数据进行标准化 
    将每一列的特征标准化，每一列表示同一类特征，类似于图片每个通道上的对应点。
'''
X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1]])

X_scaled = preprocessing.scale(X)
print(X_scaled)

# 处理数据后的均值和方差
print(X_scaled.mean(axis=0))  # [0. 0. 0.]
# print(X_scaled.mean(axis=1))

print(X_scaled.std(axis=0))  # [1. 1. 1.]

# 2.preprocessing.StandardScaler().fit(X),保存训练集中的参数， 调用fit方法，根据已有的训练数据创建一个标准化的转换器
scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
print(scaler)  # StandardScaler()

# 使用上面这个转换器去转换训练数据x,调用transform方法
print(scaler.transform(X))

print(scaler.transform([[-1, 1, 0]]))

print(scaler.mean_)  # [1.         0.         0.33333333]
print(scaler.var_)  # [0.66666667 0.66666667 1.55555556]
print(np.array([1., 2., 0]).mean())  # 1.0
print(np.array([1., 2., 0]).var())  # 0.6666666666666666
print(np.array([-1., 0, 1]).mean())
print(np.array([-1, 0, 1]).var())
print(np.array([2., 0, -1]).mean())
print(np.array([2., 0., -1]).var())