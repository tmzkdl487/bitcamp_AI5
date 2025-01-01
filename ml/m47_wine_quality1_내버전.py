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

encoder = LabelEncoder()
train_csv['type'] = encoder.fit_transform(train_csv['type'])

x = train_csv.drop(['quality'], axis=1)

y = train_csv['quality']

y = encoder.fit_transform(train_csv['quality'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    stratify=y
                                                    )

# smote = SMOTE(random_state=random_state, k_neighbors=1)
# x_train, y_train = smote.fit_resample(x_train, y_train)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name='mlogloss', 
    data_name='validation_0',
)

#2. 모델
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

#3. 컴파일, 훈련
model.fit(x_train, y_train,
         eval_set=[(x_test, y_test)],
         verbose=1,
          )

#4. 평가, 예측
results =  model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

# random_state= 9999
# accuracy_score :  0.6427272727272727

# random_state= 6363
# accuracy_score :  0.649090909090909

# random_state= 2828
# accuracy_score :  0.6527272727272727

# random_state= 8888
# accuracy_score :  0.6709090909090909

# random_state= 96
# accuracy_score :  0.6809090909090909

# SMOTE 적용 k_neighbors=3 구림.
# accuracy_score :  0.6409090909090909

