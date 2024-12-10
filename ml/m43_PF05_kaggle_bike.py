import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# Label Encoding
encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

# Feature selection
x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8)

# PolynomialFeatures 적용
pf = PolynomialFeatures(degree=2, include_bias=False)
x_train = pf.fit_transform(x_train)
x_test = pf.transform(x_test)

# 데이터 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBClassifier()  # 회귀 모델로 변경
rf = RandomForestClassifier()  # 회귀 모델로 변경
cat = CatBoostClassifier(verbose=0)  # 회귀 모델로 변경

model = StackingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],  # 'estimators'로 수정
    final_estimator=CatBoostClassifier(verbose=0),       # 'final_estimator'로 수정
    n_jobs=-1,
    cv=5,
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('Accuracy :', acc)

# Accuracy : 0.8622413427454783
