######################################################
## 그래프를 그린다. ##
# 1. value_counts -> 쓰지마
# 2. np.unique의 return_counts 쓰지마

############ 3. groupby 써, count() 써!!! ############

# plt.bar로 그린다. (quality 컬럼)

# 힌트
# 데이터 개수(y축) = 데이터갯수, 주저리 주저리...
######################################################

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

path = 'C:/ai5/_data/kaggle/wine/wine_quality/'

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
y = train_csv['quality'] -3

# quality별 데이터 개수를 groupby와 count()로 계산
grouped_quality = train_csv.groupby('quality').size()

# plt.bar로 시각화
plt.figure(figsize=(12, 6))
plt.bar(grouped_quality.index, grouped_quality.values, color='skyblue', alpha=0.8)
plt.xlabel('Quality', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of Each Quality', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

