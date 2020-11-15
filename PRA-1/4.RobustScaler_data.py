from sklearn import preprocessing
import numpy as np


x = np.array([[1, -1, 2],
              [2., 1000, 0],
              [0, 1, -1]])

scaler = preprocessing.RobustScaler()
x_scale = scaler.fit_transform(x)

print(x_scale)
print(x_scale.mean(0), x_scale.std(0))



