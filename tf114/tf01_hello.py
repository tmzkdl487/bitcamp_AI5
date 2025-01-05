import tensorflow as tf

t = tf.__version__

print(t)    # 1.14.0

## 텐서플로 설치 오류시...
# pip install protobuf==3.20
# pip install numpy==1.16

print('hello world')    # hello world

hello = tf.constant('hello world')
print(hello)    # Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))  # b'hello world'
