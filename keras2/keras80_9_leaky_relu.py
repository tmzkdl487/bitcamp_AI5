import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x):
    return np.maximum(0.01*x,x) 

x = np.arange(-5, 5, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# Leaky ReLU: 0보다 작은 경우, 작은 기울기(기본적으로 0.01)를 유지하여
# Dead Neuron(죽은 뉴런) 문제를 방지한다.
# ReLU와 유사한 계산 복잡도를 가지며, ReLU의 단점을 개선한 활성화 함수이다.
