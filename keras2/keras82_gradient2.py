import numpy as np

f = lambda x : x**2 -4*x +6
# def f(x):
#     return x**2 -4*x +6 <- 위에랑 똑같은 뜻

gradient = lambda x : 2*x -4

x = -10.0   # 초기값
epochs = 200
learning_rate = 0.1

print('epoch \t x \t f(x)') # epoch    x       f(x)
print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(0, x, f(x)))    # 00       -10.00000       146.00000

for i in range(epochs):
    x = x - learning_rate * gradient(x)
    
    print("{:02d}\t {:6.5f}\t {:6.3f}\t".format(i+1, x, f(x)))

# 200      2.00000          2.000