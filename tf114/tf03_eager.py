import tensorflow as tf

print('tf verson : ', tf.__version__)   
print('즉시실행모드 : ',tf.executing_eagerly())   

# tf verson :  1.14.0
# 즉시실행모드 :  False

# tf verson :  2.7.4
# 즉시실행모드 :  True

# tf.compat.v1.disable_eager_execution()
# print('즉시실행모드 : ',tf.executing_eagerly())   # 즉시실행모드 :  False


# tf.compat.v1.enable_eager_execution()
# print('즉시실행모드 : ',tf.executing_eagerly())   # 즉시실행모드 :  True

# 즉시실행모드 -> 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법으로 실행시킨다.
# tf.compat.v1.disable_eager_execution()    # 즉시실행모드 끈다. // 텐서플로 1.0문법 //디폴트
tf.compat.v1.enable_eager_execution()     # 즉시실행모드 켠다. // 텐서플로 2.0사용가능

hello = tf.constant('Hello World')

sess = tf.compat.v1.Session()
print(sess.run(hello))

# 가상환경  즉시실행모드      사용가능
# 1.14.0    desable(디폴트)  b'Hello World'
# 1.14.0    enable           에러
# 2.7.4     desable(디폴트)  b'Hello World'
# 2.7.4     enable           에러

'''
Tensor 1은 '그래프연산' 모드
Tensor 2는 '즉시실행' 모드

tf.compat.v1.enable_eager_execution()  # 즉시 실행모드 켜
            -> Tensor 2의 디폴트
            
tf.compat.v1.disable_eager_execution()   # 즉시 실행모드 꺼            
                                            -> 그래프 연산모드로 돌아간다.
                                            -> Tensor 1 코드를 쓸 수 있다.

tf.executing_eagerly()  # True면 즉시 실행모드, -> Tensor 2 코드만 써야 한다.
                          False면 그래프 연산모드 -> Tensor 1 코드를 쓸 수 있다.                                           
'''