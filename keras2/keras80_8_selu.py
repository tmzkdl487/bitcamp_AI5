import numpy as np
import matplotlib.pyplot as plt

def selu(x, alpha=1.67326, scale=1.0507):
    return np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# SELU (Scaled Exponential Linear Unit): 
# SELU는 입력 값에 따라 두 가지 방식으로 출력값을 계산한다:
# - x > 0: y = scale * x
# - x <= 0: y = scale * alpha * (exp(x) - 1)
# 여기서 scale(λ) ≈ 1.0507, alpha(α) ≈ 1.67326로 고정된 값이 사용된다.
# SELU의 주요 특징:
# 1. 자체 정규화(Self-Normalizing): 활성화 함수의 출력값이 일정한 평균과 분산을 유지하도록 설계됨.
# 2. 음수 영역에서도 매끄러운 동작: ReLU와 달리 음수 입력을 부드럽게 처리해 정보 손실을 방지.
# 3. 깊은 신경망에서 효과적: 층을 깊게 쌓아도 분산이 안정적으로 유지되어 학습 성능이 향상됨.
# SELU는 특히 Batch Normalization 없이도 안정적인 학습이 가능하도록 설계된 활성화 함수이다.
