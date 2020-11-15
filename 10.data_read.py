import numpy as np
import sklearn
import os
from sklearn.datasets import load_iris, load_wine

iris = load_iris()
# print(iris)

wine_file = open("./data/wine.data")
wine = wine_file.readlines()
x = []
y = []
for L in wine:
    str = L.strip().split(",")
    _y = np.float64(str[0])
    _x = np.float64(str[1:])
    x.append(_x)
    y.append(_y)

# print(x)
x = np.stack(x)
print(x)
y = np.stack(y)
print(y)

