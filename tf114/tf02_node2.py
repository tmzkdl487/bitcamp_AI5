import tensorflow as tf

nodel = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습
# 덧셈: node3
# 뺄셈: node4
# 곱셈: node5
# 나눗셈: node6

node3 = tf.add(nodel, node2)# 덧셈
node4 = tf.subtract(nodel, node2)  # 뺄셈
node5 = tf.multiply(nodel, node2)  # 곱셈
node6 = tf.divide(nodel, node2)  # 나눗셈

sess = tf.compat.v1.Session()
print("덧셈 (node3):", sess.run(node3))
print("뺄셈 (node4):", sess.run(node4))
print("곱셈 (node5):", sess.run(node5))
print("나눗셈 (node6):", sess.run(node6))