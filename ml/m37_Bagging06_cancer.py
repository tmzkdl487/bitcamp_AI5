import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # <- 분류 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    random_state=4444,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
model = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators =100,
                          n_jobs = -1,
                          random_state=4444, 
                        #   bootstrap=True,   # 디폴트, 중복 허용
                          bootstrap=False     # 중복허용 안함.
                          )

# model = LogisticRegression()

# model = BaggingClassifier(LogisticRegression(),
#                           n_estimators =100,
#                           n_jobs = -1,
#                           random_state=4444, 
#                           bootstrap=True,   # 디폴트, 중복 허용
#                         #   bootstrap=False     # 중복허용 안함.
#                           )


# model = BaggingClassifier(RandomForestClassifier(),
#                           n_estimators =100,
#                           n_jobs = -1,
#                           random_state=4444, 
#                         #   bootstrap=True,   # 디폴트, 중복 허용
#                           bootstrap=False     # 중복허용 안함.
#                           )

# model = RandomForestClassifier()

model = BaggingClassifier(RandomForestClassifier(),
                          n_estimators =100,
                          n_jobs = -1,
                          random_state=4444, 
                        #   bootstrap=True,   # 디폴트, 중복 허용
                          bootstrap=False     # 중복허용 안함.
                          )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score: ', acc)

# 디시전
# 최종점수 :  0.956140350877193
# acc_score:  0.956140350877193

# 디시전 배깅 부트스트랩 투루
# 최종점수 :  0.9385964912280702
# acc_score:  0.9385964912280702

# 디시전 배깅 부트스트랩 펄스
# 최종점수 :  0.9649122807017544
# acc_score:  0.9649122807017544

# 로지스틱
# 최종점수 :  0.9736842105263158
# acc_score:  0.9736842105263158

# 로지스틱 배깅, 부두스트랩 투루
# 최종점수 :  0.9649122807017544
# acc_score:  0.9649122807017544

# 로지스틱 배깅, 부투스트랩  펄스
# 최종점수 :  0.9736842105263158
# acc_score:  0.9736842105263158

# 랜포
# 최종점수 :  0.956140350877193
# acc_score:  0.956140350877193

# 랜포배깅, 부투스트랩  투루
# 최종점수 :  0.9649122807017544
# acc_score:  0.9649122807017544

# 랜포배깅, 부투스트랩  펄스
# 최종점수 :  0.956140350877193
# acc_score:  0.956140350877193