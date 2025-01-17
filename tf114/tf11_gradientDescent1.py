import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

#2. 모델
hypothesis = x * w

#3-1. 컴파일 ## model.compile(loss='mse', optimizer ='sad')
loss = tf.reduce_mean(tf.square(hypothesis - y))

############### 옵티마이저 ###################
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   # <- Adam쓰고 싶으면 GradientDescentOptimizer을 Adam으로 바꾸면 됨.
# train = optimizer.minimize(loss)

lr = 0.1
gradient = tf.reduce_mean(( x*w - y ) * x)

descent = w - lr * gradient

train = w.assign(descent)

############### 옵티마이저 끝 ###################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
for step in range(31):
    _, loss_v, w_v = sess.run([train, loss, w], feed_dict={x: x_train, y:y_train})
    print(step, '\t', loss_v, '/t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()

print("---------------------- W history --------------------------")
print(w_history)
print("---------------------- loss history --------------------------")
print(loss_history)

plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.show()
    
