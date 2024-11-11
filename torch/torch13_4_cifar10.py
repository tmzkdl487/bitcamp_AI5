import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용자DEVICE:', DEVICE)

path = 'C:/ai5/_data/torch이미지/'
train_dataset = CIFAR10(path, train=True, download=False)
test_dataset = CIFAR10(path, train=False, download=False)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets

# print(x_train.shape, len(y_train)) 
# (50000, 32, 32, 3) 50000

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)

x_train = x_train.reshape(-1, 32*32*3)
x_test = x_test.reshape(-1, 32*32*3)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

#2. 모델
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.hidden_layer1 = nn.Sequential(
           nn.Linear(num_features, 128),
           nn.ReLU(),
           nn.Dropout(0.5)
       )
        self.hidden_layer2 = nn.Sequential(
           nn.Linear(128, 128),
           nn.ReLU(),
           nn.Dropout(0.5)
       )
        self.hidden_layer3 = nn.Sequential(
           nn.Linear(128, 64),
           nn.ReLU(),
           nn.Dropout(0.5)
       )
        self.hidden_layer4 = nn.Sequential(
           nn.Linear(64, 64),
           nn.ReLU(),
           nn.Dropout(0.5)
       )
        self.hidden_layer5 = nn.Sequential(
           nn.Linear(64, 32),
           nn.ReLU(),
           nn.Dropout(0.5)
       )
        self.output_layer = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x

model = DNN(32*32*3).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4) # 0.0001

def train(model, criterion, optimizer, loader):
    # moel.train()
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()   # for문 안에 넣어야됨. 배치 단위로 기울기를 0으로 만들어야되기 때문.
        hypothesis = model(x_batch) # y = xw+b
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()  # 기울기 값 계산
        optimizer.step() # 가중치 갱신
        
        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_acc += acc
        
    return epoch_loss / len(loader), epoch_acc / len(loader) 

def evaluate(model, criterion, loader):
    model.eval()
    
    epoch_loss = 0 
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader :
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
        
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            epoch_acc += acc.item()
            
    return epoch_loss / len(loader), epoch_acc / len(loader)
#  loss, acc = model.evaluate(x_test, y_test)

epochs = 20
for epoch in range(1, epochs +1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print('epoch:{}, loss:{:.4f}, acc:{:.3f}, val_loss:{:.4f}, val_acc:{:.3f}'.format(
         epoch, loss, acc, val_loss, val_acc
     ))

#4. 평가, 예측
loss, acc = evaluate(model, criterion, test_loader)
print('최종 loss:', loss, '최종acc:', acc)
print('최종 loss:', round(loss), '최종acc:', round(acc))

# 최종 loss: 1.506813501778502 최종acc: 0.4610623003194888
# 최종 loss: 1 최종acc: 0

# 최종 loss: 3.5824059099435046 최종acc: 0.16024361022364217
# 최종 loss: 4 최종acc: 0