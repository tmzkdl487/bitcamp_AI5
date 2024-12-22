# m47_wine_quality1_선생님버전.py 카피

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
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
y = train_csv['quality']-3

# print(x.shape)  # (5497, 12)
# print(y.shape)  # (5497,)

# y = y-3 # 왜 라벨인코더 썻어? ㅋㅋㅋ

########################################################################################
# [실습] y의 클래스를 7개에서 5 ~ 3개로 줄여서 성능을 비교.
########################################################################################
y = y.copy()    # 메모리 안전빵 알아서 참고 하고.

## 힌트: for문 돌리면 되겠지?

'''
for i, v in enumerate(y):
    if v <=4:
        y[i] = 0
    elif v == 5:
    # elif v == 6:
         y[i] = 1
    elif v == 6:
        y[i] = 1
    else:
        y[i] = 2
'''

# 이거보다 위에가 더 나음.
# for i, v in enumerate(y):
#     if v <=4:
#         y[i] = 0
#     # elif v == 5:
#     elif v == 6:
#         y[i] = 1
#     elif v == 7:
#         y[i] = 1
#     else:
#         y[i] = 2

print(y.value_counts().sort_index())
# 0     212
# 1    1788
# 2    3497
# Name: quality, dtype: int64

# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)

# 데이터 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(random_state=random_state)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
results = model.score(x_test, y_test)
print("model.score:", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score:", acc)
print('F1 : ', f1_score(y_test, y_predict, average='macro'))

###### 원판 (7개)
# ACC : 0.60272727272728

###### y클래스 변경 후 (3개 일때.)
# accuracy_score: 0.8354545454545454
# F1 :  0.5528689039718451