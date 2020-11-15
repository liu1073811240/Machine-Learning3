import numpy as np

file = open("./data/test.txt")

str = file.read()  # 读取全部数据
# str = file.readline()  # 只读取一行数据
# str = file.readlines()  # 读取所有行数据

# str = np.loadtxt("./data/test.txt")
print(str)

