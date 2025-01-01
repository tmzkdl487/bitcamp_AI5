# [맹그러봐] 스모트 써서 비교해봐!!!
# y 클래스 7개까지 그대로

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

random_state= 96

#1. 데이터
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
y = train_csv['quality'] -3

# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_state, stratify=y)

smote = SMOTE(random_state=random_state, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

# 데이터 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name='mlogloss', 
    data_name='validation_0',
)

# 2. 모델
model = XGBClassifier(
    n_estimators = 500,
    learning_rate=0.1,
    max_depth = 6,
    gamma = 0,
    min_child_weight= 0,
    subsample= 0.8,
    colsample_bytree=0.8,
    callbacks=[early_stop], 
    random_state=random_state,
    )

# 3. 훈련
model.fit(x_train, y_train,
         eval_set=[(x_test, y_test)],
         verbose=1,
          )

# 4. 평가
results = model.score(x_test, y_test)
print("model.score:", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score:", acc)
print('F1 : ', f1_score(y_test, y_predict, average='macro'))

# model.score: 0.7018181818181818
# accuracy_score: 0.7018181818181818
# F1 :  0.5146284798529924