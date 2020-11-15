from sklearn import preprocessing
import numpy as np

enc = preprocessing.OneHotEncoder()
enc1 = preprocessing.OneHotEncoder(sparse=False)

ans = enc.fit_transform([[0], [1], [2], [1]])
ans1 = enc1.fit_transform([[0], [1], [2], [1]])

print(ans)
print(ans1)


# 利用numpy实现one-hot编码，比如[5, 2, 8, 6]
y = np.array([5, 2, 8, 6])

print(np.eye(10))
print(np.eye(10)[y])






