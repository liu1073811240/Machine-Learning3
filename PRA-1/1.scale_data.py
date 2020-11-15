from sklearn import preprocessing
import numpy as np

'''1.preprocessing.scale(x), 可以直接将给定数据进行标准化
    将每一列的特征标准化，每一列表示同一类特征, 类似图片每个通道上的对应点 

'''
X = np.array([[1, -1, 2.],
              [2., 0, 0],
              [0, 1, -1]])

X_scaled = preprocessing.scale(X)
print(X_scaled)

# 处理数据后的均值和方差
print(X_scaled.mean(axis=0))

print(X_scaled.std(axis=0))

# 2.preprocessing.StandardScaler().fit(X)
scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
print(scaler)

# 使用上面这个转换器训练数据x，调用transform方法
print(scaler.transform(X))

print(scaler.transform([[-1, 1, 0]]))

print(scaler.mean_)
print(scaler.var_)

print(np.array([1., 2., 0]).mean())
print(np.array([1, 2, 0.]).var())

print(np.array([-1, 0, 1]).mean())
print(np.array([-1, 0, 1]).var())
print(np.array([2, 0, -1]).mean())
print(np.array([2, 0, -1]).var())






