import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # <- 분류 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import PolynomialFeatures

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    random_state=1199,  # 1199도 스태킹 0.99나옴.
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
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

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 ACC : ', accuracy_score(y_test, y_pred))

# model.score :  0.9912280701754386
# 스태킹 ACC :  0.9912280701754386 