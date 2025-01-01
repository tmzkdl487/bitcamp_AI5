# [실습] 얼리스타핑 적용하려면 어떻게 하면 될까요???
# 1. 최소값을 넣을 변수를 하나, 카운트할 변수 하나 준비!!!
# 2. 다음 에포에 값과 최소값을 비교. 최소값이 갱신되면 
#    그 변수에 최소값을 넣어주고, 카운트변수 초기화
# 3. 갱신이 안되면 카운트 변수 ++1
#    카운트 변수가 내가 원하는 얼리스타핑 갯수에 도달하면  for문을 stop

x = 10

y = 10

w = 0.001

lr = 0.001

epochs = 1000

# Early Stopping 관련 변수
min_loss = float('inf')  # 최소 손실값 초기화
patience = 5  # 얼리 스탑핑 기준: 몇 번 동안 갱신이 없을 때 멈출지
no_improve_count = 0  # 갱신되지 않은 횟수
threshold = 1e-6  # 손실 차이에 따른 임계값

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y)**2  # mse
    
    print(f"Epoch: {i + 1}, Loss: {round(loss, 6)}, Predict: {round(hypothesis, 4)}")
    
    # Early Stopping 체크
    if abs(loss - min_loss) > threshold:  # 손실 값이 threshold 이상으로 줄어들 경우에만 갱신
        min_loss = loss        # 최소 손실 갱신
        no_improve_count = 0   # 카운트 초기화
    else:
        no_improve_count += 1  # 갱신되지 않으면 카운트 증가
    
    # 얼리 스탑핑 조건 충족 시 루프 종료
    if no_improve_count >= patience:
        print(f"Early stopping triggered at epoch {i + 1}")
        break
    
    # 가중치 업데이트
    up_predict = x * (w +lr)
    up_loss = (y - up_predict) **2
    
    down_predcit = x * (w - lr)
    down_loss = (y - down_predcit) **2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr