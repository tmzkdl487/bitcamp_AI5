# 얼리스타핑 만드는 법
# 1 최소값을 넣을 변수 하나, 카운트할 변수 하나를 준비한다
# 2 다음 epoch에서 값과 최소값을 비교
#   최소값이 갱신되면 그 변수에 최소값을 넣어주고 카운트 변수 초기화
# 3 카운트 변수가 내가 원하는 얼리스타핑 갯수에 도달하면 for문 stop

x = 10
y = 10
w = 0.005
lr = 0.001 # lr too large -> pingpong, lr too small -> not update weight
epochs = 100000

maximum_value = -99999999999
patience = 100

for i in range(epochs):
    hypothesis = x * w

    loss = (hypothesis - y) ** 2 # mse

    print('Loss :', round(loss, 4), '\tPredict :', round(hypothesis, 4), '\ti :', round(i, 4), '\tw :', round(w, 4), '\t', maximum_value, '\tpatience :', patience)

    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2

    if (up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr

    if patience == 1:
        break

    if maximum_value < round(w, 4):
        maximum_value = round(w, 4)

        patience = 100
    else:
        patience -= 1