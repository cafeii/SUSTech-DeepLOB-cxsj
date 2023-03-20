import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:,self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

root_train = '../data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training'
root_test = '../data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing'
train_data_path = root_train + '/Train_Dst_NoAuction_ZScore_CF_7.txt'
test_data_path1 = root_test + '/Test_Dst_NoAuction_ZScore_CF_7.txt'
test_data_path2 = root_test + '/Test_Dst_NoAuction_ZScore_CF_8.txt'
test_data_path3 = root_test + '/Test_Dst_NoAuction_ZScore_CF_9.txt'

dec_traindata = np.loadtxt(train_data_path)
dec_test1 = np.loadtxt(test_data_path1)
dec_test2 = np.loadtxt(test_data_path2)
dec_test3 = np.loadtxt(test_data_path3)

dec_train = dec_traindata[:, :int(np.floor(dec_traindata.shape[1]*0.8))]
dec_val = dec_traindata[:, int(np.floor(dec_traindata.shape[1]*0.8)):]

dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

batch_size = 32

dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
dataset_val = Dataset(data=dec_val, k=4, num_classes=3, T=100)
dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=100)

train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

size = len(dataset_train) // 7
train_size = int(size * 0.6)
val_size = int(size * 0.2)

dec_data = dec_traindata[:, :size]
dec_train = dec_traindata[:, :train_size]
dec_val = dec_traindata[:, train_size:train_size+val_size]
dec_test = dec_traindata[:, train_size+val_size:]

tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)

class deeplob(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.conv1 = nn.Sequential( # 144 * 40
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) # 16 * 138 * 20

        self.conv2 = nn.Sequential( # 16 * 142 * 20
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) # 16 * 132 * 10
        
        self.conv3 = nn.Sequential( # 16 * 132 * 10
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 10)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) # 16 * 126 * 1

        self.incept1 = nn.Sequential( # 16 * 126 * 1
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # 32 * 124 * 1
        
        self.incept2 = nn.Sequential( # 32 * 124 * 1
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # 32 * 120 * 1

        self.incept3 = nn.Sequential( # 32 * 120 * 1
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1,1),padding=(1,0)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # 32 * 38 * 1

        self.lstm = nn.LSTM(input_size=96, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, length)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(device)
        c0 = torch.zeros(1, x.size(0), 64).to(device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_incept1 = self.incept1(x)
        x_incept2 = self.incept2(x)
        x_incept3 = self.incept3(x)  
        
        x = torch.cat((x_incept1, x_incept2, x_incept3), dim=1)
        
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)
        
        return forecast_y

model = deeplob(dataset_train.num_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

train_losses = np.zeros(epochs)
test_losses = np.zeros(epochs)

for it in tqdm(range(epochs)):
    st = datetime.now()
    model.train()
    train_loss = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_loss = np.mean(train_loss)
    model.eval()
    test_loss = []
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)      
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss.append(loss.item())
    test_loss = np.mean(test_loss)
    
    train_losses[it] = train_loss
    test_losses[it] = test_loss

    dt = datetime.now() - st
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}')

torch.save(model, './a_very_good_model')

#train_losses
#test_losses
with open('train_losses.txt', mode='w',encoding='utf-8') as file:
    file.write(train_losses)

with open('test_losses.txt', mode='w',encoding='utf-8') as file:
    file.write(test_losses)