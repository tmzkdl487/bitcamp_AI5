import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]   # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                     # (6, 1)

###################################################################
####  [실습] 기냥 한번 맹그려봐!!!!
###################################################################

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2, 1], name = 'weight'))
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델구성
hypothesis = tf.sigmoid(tf.matmul(x,w) + b) # 행렬 곱

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)  
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(2001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epoch % 200 == 0:
        print(epoch, 'loss:', loss_val, '\n', hy_val)
    
#4. 예측
y_predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)  # tf.cast = 자료형을 바꿔주는것 (FFFTTT를 float형인 0.0.0.1.1.1.로 바꿔주겠다!) // False면 0, True면 

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype = tf.float32)) # tf.equal: 비교역할 / tf.cast: 동일하면 1 동일하지 않으면 0 반환

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})    # 사람이 해석할 수 있게 변환됨
print("=========================")
print("예측값: \n", hy_val)
print("예측결과: \n", pred)
print("Accuracy: ", acc)

sess.close() 

# 0.5 이상이면 1 // 0.5 이하면 0으로 판단!!!!!!  


