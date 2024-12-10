import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression # <- 분류 모델, 
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

# PCA 적용
pca = PCA(n_components=10)  # 필요한 차원 수로 설정
x = pca.fit_transform(x)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    random_state=4444,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    # stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

model = StackingRegressor(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],  # 'estimators'로 수정
    final_estimator=CatBoostRegressor(verbose=0),       # 'final_estimator'로 수정
    n_jobs=-1,
    cv=5,
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 ACC : ', r2_score(y_test, y_pred))
