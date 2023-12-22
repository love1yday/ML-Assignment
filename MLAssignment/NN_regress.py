import datetime

# 个人信息
student_id = "2010040116"
name = "zcg" 

# 获取系统时间
current_time = datetime.datetime.now()


def myfun(x):
    '''目标函数
    input:x(float):自变量
    output:函数值'''
    return 2 + 1 * x + 4 * x**2 + 1 * x**3+ 1 * x**4 + 6 * x**5 

import numpy as np
x = np.linspace(-3,3, 20)
y = myfun(x) + np.random.random(size=len(x)) * 100 - 50
yy = y.copy()

# 数据标准化
miny = min(y)
maxy = max(y)
def standard(y, miny, maxy):
    step = maxy - miny
    for i in range(len(y)):
        y[i] = (y[i] - miny)/step

standard(y, miny, maxy)

# 数据反标准化
def invstandard(y, miny, maxy):
    step = maxy - miny
    for i in range(len(y)):
        y[i] = miny + y[i]*step
    
import tensorflow as tf

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(1,), 
                          kernel_initializer='random_uniform', bias_initializer='zeros'),
    tf.keras.layers.Dense(10, activation='sigmoid', 
                          kernel_initializer='random_uniform', bias_initializer='zeros'),
    tf.keras.layers.Dense(10, activation='sigmoid', 
                          kernel_initializer='random_uniform', bias_initializer='zeros'),
    tf.keras.layers.Dense(1, activation='sigmoid',
                          kernel_initializer='random_uniform', bias_initializer='zeros')
])

# 编译模型
model.compile(optimizer='adam',
              loss='mean_squared_error')
# 训练模型
model.fit(x, y, batch_size=20, epochs=1000, verbose=1)
# 打印模型摘要
model.summary()

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
plt.rc('font', family='SimHei', size=13)
plt.scatter(x, yy, color="black", linewidth=2)
x1 = np.linspace(-3, 3, 100)
y0 = myfun(x1)
plt.plot(x1, y0, color="red", linewidth=1)

# 使用训练好的模型预测并反标准化
y1 = model.predict(x1)
invstandard(y1, miny, maxy)
plt.plot(x1, y1, "b--", linewidth=1)

# 输出个人信息和系统时间
print(f"Student ID: {student_id}")
print(f"Name: {name}")
print(f"Current Time: {current_time}")
# 显示图像
plt.show()