import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

# PolynomialFeatures 적용
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)

# PCA 적용
pca = PCA(n_components=10)  # 필요한 차원 수로 설정
x = pca.fit_transform(x)

# Train-test split
random_state = 777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1199, shuffle=True, train_size=0.8, stratify=y
)

# 데이터 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

model = StackingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 및 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 ACC : ', accuracy_score(y_test, y_pred))

# model.score :  0.9736842105263158
# 스태킹 ACC :  0.9736842105263158