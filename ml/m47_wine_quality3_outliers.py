# [실습]
# 1. 아웃라이어 확인
# 2. 아웃라이어 처리
# 3. 47_1 이느 47_2 든 수정해서 맹그러

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import warnings
warnings.filterwarnings('ignore')

path = 'C:/ai5/_data/kaggle/wine/wine_quality/'

# [맹그러봐 ] : y는 quality

#1. 데이터
random_state= 11

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

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,    # percentile 백분율
                                               [25, 50, 75])
    
    print("1사분위 : ", quartile_1)  # 4.0
    print("q2 : ", q2)              # 중위값 : 7.0
    print("3사분위 : ", quartile_3)  # 10.0
    iqr = quartile_3 - quartile_1   # 10.0 - 4.0 = 6.0
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound)), iqr
    
outliers_loc, iqr = outliers(x)
print("이상치의 위치 : ", outliers_loc)

# plt.figure(figsize=(12, 6))
# x.boxplot()
# plt.xticks(rotation=45, fontsize=10)
# plt.axhline(iqr, color='red', label='TQR')
# plt.legend()
# plt.title("Boxplot with Column Names")
# plt.show()

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

path = 'C:/ai5/_data/kaggle/wine/wine_quality/'

# 1. 데이터 로드
random_state = 11
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

# Label Encoding
type_le = LabelEncoder()
train_csv['type'] = type_le.fit_transform(train_csv['type'])

# Feature와 Target 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality'] - 3  # y를 0부터 시작하도록 조정

# 이상치 제거 함수
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# 전체 데이터에 이상치 제거 적용
x_cleaned = remove_outliers(x)
y_cleaned = y[x_cleaned.index]

print("이상치 제거 후 데이터 크기 (전체):", x_cleaned.shape, y_cleaned.shape)

# 특정 열(`total sulfur dioxide`)에 대해 추가적인 이상치 제거 적용
def remove_outliers_column(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

x_cleaned = remove_outliers_column(x_cleaned, 'total sulfur dioxide')
y_cleaned = y_cleaned[x_cleaned.index]

print("`total sulfur dioxide` 이상치 제거 후 데이터 크기:", x_cleaned.shape, y_cleaned.shape)

# 박스플롯 그리기 (이상치 제거 후 확인)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
x_cleaned.boxplot()
plt.xticks(rotation=45, fontsize=10)
plt.title("Boxplot After Outlier Removal")
plt.show()

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_cleaned, y_cleaned, random_state=random_state, train_size=0.8, stratify=y_cleaned
)

# SMOTE 적용
smote = SMOTE(random_state=random_state, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

# 데이터 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 정의
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    random_state=random_state
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
results = model.score(x_test, y_test)
print("model.score:", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score:", acc)

# model.score: 0.6444141689373297
# accuracy_score: 0.6444141689373297