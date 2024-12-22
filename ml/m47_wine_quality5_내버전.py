# m47_wine_quality1_선생님버전.py 카피

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

path = 'C:/ai5/_data/kaggle/wine/wine_quality/'

# [맹그러봐 ] : y는 quality

#1. 데이터
random_state= 96

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

# print(train_csv) 
# [5497 rows x 13 columns]

# print(train_csv['quality'].value_counts().sort_index())
# 3      26
# 4     186
# 5    1788
# 6    2416
# 7     924
# 8     152
# 9       5
# Name: quality, dtype: int64

le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
# print(aaa)          # [1 0 1 ... 1 1 1]
# print(type(aaa))    # <class 'numpy.ndarray'>
# print(aaa.shape)    # (5497,)

train_csv['type'] = aaa

# print(le.transform(['red', 'white']))   # [0 1] <- red가 0, white가 1

# print(train_csv.describe())   # 데이터프레임의 수치형 열에 대해 기본 통계 정보를 요약
# print(train_csv.info())       #  데이터프레임의 전체 구조와 요약 정보를 보여줌

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

# print(x.shape)  # (5497, 12)
# print(y.shape)  # (5497,)

# y = y-3 # 왜 라벨인코더 썻어? ㅋㅋㅋ

########################################################################################
# [실습] y의 클래스를 7개에서 5 ~ 3개로 줄여서 성능을 비교.
########################################################################################
y = y.copy()    # 메모리 안전빵 알아서 참고 하고.

## 힌트: for문 돌리면 되겠지?

# 클래스 개수를 줄이는 함수 정의
def reduce_classes(y, num_classes):
    if num_classes == 5:
        y = y.apply(lambda x: 3 if x <= 4 else 4 if x == 5 else 5 if x == 6 else 6 if x == 7 else 7)
    elif num_classes == 4:
        y = y.apply(lambda x: 3 if x <= 4 else 4 if x <= 5 else 5 if x <= 6 else 6)
    elif num_classes == 3:
        y = y.apply(lambda x: 3 if x <= 4 else 4 if x <= 6 else 5)
    else:
        raise ValueError("num_classes should be 5, 4, or 3.")
    return y

# 결과 저장용 딕셔너리
results = {}

# 클래스 개수를 5, 4, 3으로 줄여가며 성능 비교
for num_classes in [5, 4, 3]:
    y_reduced = reduce_classes(y, num_classes)
    
    # 클래스 라벨을 0부터 시작하도록 조정
    y_reduced = y_reduced - y_reduced.min()

    # 데이터 나누기
    x_train, x_test, y_train, y_test = train_test_split(x, y_reduced, test_size=0.2, random_state=random_state, stratify=y_reduced)

    # 데이터 스케일링
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 모델 정의 및 학습
    model = XGBClassifier(random_state=random_state)
    model.fit(x_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # 결과 저장
    results[num_classes] = acc
    
    # 결과 출력
for num_classes, acc in results.items():
    print(f"Number of classes: {num_classes}, Accuracy: {acc:.4f}")

# 결과 출력 및 성능 그래프 그리기
num_classes_list = list(results.keys())
accuracy_list = list(results.values())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(num_classes_list, accuracy_list, marker='o', linestyle='-', markersize=8)
plt.title('Accuracy vs Number of Classes', fontsize=14)
plt.xlabel('Number of Classes', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(num_classes_list)  # x축 클래스 수 설정
plt.grid(True)
plt.show()