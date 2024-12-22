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

y = y-3 # 왜 라벨인코더 썻어? ㅋㅋㅋ

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True, 
                                                    random_state=random_state,
                                                    train_size=0.8,
                                                    stratify=y
                                                    )

parameters = {
            'n_estimators' : 500,
            'learning_rat': 0.1,
            'max_depth' : 6,
            'gamma' : 0,
            'min_child_weight': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
}

#2. 모델
model = XGBClassifier(**parameters, n_jobs=-1)
model.set_params(early_stopping_rounds=500,
                eval_metric='merror')

#3. 훈련
model.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            verbose=1,
            )

results = model.score(x_test, y_test)
print('model.score : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

