import tensorflow as tf

#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. 모델 구성
# y = wx + b => y = xw + b  # 이제는 말할 수 있다.

hypothesis = x * w + b

# 3-1. 컴파일 
# model.compoile (loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)   # <- Adam쓰고 싶으면 GradientDescentOptimizer을 Adam으로 바꾸면 됨.
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화

# model.fit()
epochs = 1001
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:    
        print(step, sess.run(loss), sess.run(w), sess.run(b))   # <- verbose임.
sess.close()    # 메모리 많이 차이하니까. 원래 sess.run하고 sess.close 해줘야됨.




    
    

    