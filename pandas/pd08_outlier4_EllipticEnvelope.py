import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T

### for 문 만들어서 돌려봐 ###

from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.3)
outliers = EllipticEnvelope()

# 각 열에 대해 이상치 탐지 수행
for col_idx in range(aaa.shape[1]):
    print(f"\n열 {col_idx+1} 분석:")
    col_data = aaa[:, col_idx].reshape(-1, 1)  # 열 데이터를 2D 형태로 변환

    outliers.fit(col_data)  # 모델 학습
    results = outliers.predict(col_data)  # 이상치 탐지

    print("결과: ", results)
    print("이상치 위치: ", np.where(results == -1)[0])

# 열 1 분석:
# 결과:  [-1  1  1  1  1  1  1  1  1  1  1  1 -1]
# 이상치 위치:  [ 0 12]

# 열 2 분석:
# 결과:  [ 1  1  1  1  1  1 -1  1  1 -1  1  1  1]
# 이상치 위치:  [6 9]