import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

# 创建数据集
x = np.array(list(range(1, 11))).reshape(-1, 1)
print(x)

y = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9., 9.05]).reshape(-1, 1)

# 回归模型拟合数据
model1 = DecisionTreeRegressor(max_depth=1)
model2 = DecisionTreeRegressor(max_depth=15)
model3 = linear_model.LinearRegression()

model1.fit(x, y)
model2.fit(x, y)
model3.fit(x, y)

# 预测
X_test = np.arange(0.0, 10.0, 0.01).reshape(-1, 1)
print(X_test)

y1 = model1.predict(X_test)
y2 = model2.predict(X_test)
y3 = model3.predict(X_test)

plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y1, color="cornflowerblue", label="max_depth=1", linewidth=2)
plt.plot(X_test, y2, color="yellowgreen", label="max_depth=3", linewidth=2)
plt.plot(X_test, y3, color="red", label="linear regression", linewidth=2)

plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()






