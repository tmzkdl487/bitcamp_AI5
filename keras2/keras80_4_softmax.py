import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 5)
print(x)    # [1 2 3 4]

def softmax(x) :
    return np.exp(x) / np.sum(np.exp(x))

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))

y = softmax(x)

ratio = y
labels = y

plt.pie(ratio, labels, shadow = True, startangle=90)
plt.show()