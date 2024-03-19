# %% 
import os

import pandas as pd
import pickle

base_load_path = '/home/user/Documents/oy6uns/Source/0. Dicts'

pickle_files = ['train_X.pkl', 'train_y.pkl', 'test_X.pkl', 'test_y.pkl']

loaded_dicts = {}

for pickle_file in pickle_files:
    pickle_file_path = os.path.join(base_load_path, pickle_file)
    
    # 저장된 딕셔너리를 load해준다. 
    with open(pickle_file_path, 'rb') as file:
        loaded_dicts[pickle_file.replace('.pkl', '')] = pickle.load(file)

# 'loaded_dicts'에 각 딕셔너리를 저장해준다. 
train_X = loaded_dicts['train_X']
train_y = loaded_dicts['train_y']
test_X = loaded_dicts['test_X']
test_y = loaded_dicts['test_y']
# %%
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Flatten the train_X data
X = []
for region in train_X.keys():
    for file in train_X[region].keys():
        # DataFrame에서 불필요한 열을 제거하고 값들을 추출
        data = train_X[region][file].drop(columns=['Unnamed: 0']).values
        normalized_data = scaler.fit_transform(data)

        # 정규화된 데이터를 1차원 배열로 변환하여 리스트에 추가
        X.append(normalized_data.flatten())

# Prepare the train_y labels for the four categories
y_memberNo = []
y_jobType = []
y_houseType = []
y_houseArea = []

for region in train_y.keys():
    for file in train_y[region].keys():
        # Append the label for each category to its respective list
        y_memberNo.append(train_y[region][file].query("type == 'memberNo'")['value'].iloc[0])
        y_jobType.append(train_y[region][file].query("type == 'jobType'")['value'].iloc[0])
        y_houseType.append(train_y[region][file].query("type == 'houseType'")['value'].iloc[0])
        y_houseArea.append(train_y[region][file].query("type == 'houseArea'")['value'].iloc[0])

# Convert lists to numpy arrays
X = np.array(X)
y_memberNo = np.array(y_memberNo)
y_jobType = np.array(y_jobType)
y_houseType = np.array(y_houseType)
y_houseArea = np.array(y_houseArea)

target_y = y_memberNo
# %%
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

random.seed(42)

# train, valid 나누기
X_train, X_val, y_train, y_val = train_test_split(torch.Tensor(X), torch.Tensor(target_y), test_size=0.2)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = (y-1).long()  # classify 시에 오류 발생 막기 위한 정수형 변환
        '''CrossEntropyLoss를 사용하기 위해 1, 2, 3의 Label값에 1을 빼주어 0, 1, 2로 매핑'''
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# %%
class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

model = MLPClassifier(input_size=X_train.shape[1], output_size=torch.unique(y_train).shape[0])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
# %%
train_loss = []
train_accuracy = []
valid_loss = []
valid_accuracy = []
_epoch = 500
patience = 10  # 연속으로 손실이 개선되지 않는 에포크 허용 횟수
counter = 0   # 개선되지 않는 에포크를 카운트하는 변수
prev_loss = float('inf')  # 최소 손실을 저장하는 변수
best_val_acc = 0  # best model을 선별하기 위한 변수

# CUDA 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 모델을 GPU로 이동
model = model.to(device)

# Train
for epoch in range(_epoch):
    correct = 0
    total = 0
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        output = model(data)
        loss = criterion(output, labels)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_epoch_accuracy = 100 * correct / total
    train_loss.append(loss.item())
    train_accuracy.append(train_epoch_accuracy)
    print(f'Epoch [{epoch+1}/{_epoch}], Loss: {loss.item():.4f}, Accuracy: {train_epoch_accuracy}%')

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            loss = criterion(output, labels)

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_epoch_accuracy = 100 * correct / total
    valid_loss.append(loss.item())
    valid_accuracy.append(valid_epoch_accuracy)
    print(f'validation set Loss: {loss.item():.4f}, Accuracy: {valid_epoch_accuracy}%')

    if valid_epoch_accuracy > best_val_acc:
        best_val_acc = valid_epoch_accuracy
        # 모델 상태 저장
        torch.save(model.state_dict(), 'memberNo_best_model.pth')
        print(f'Model saved at epoch {epoch+1} with validation acc: {best_val_acc}%')
        best_model_epoch = epoch+1
        best_model_train_accuracy = train_epoch_accuracy
        best_model_valid_accuracy = valid_epoch_accuracy

    # Early Stopping Check
    if loss.item() > prev_loss:
        counter += 1
        if counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    else:
        counter = 0

    prev_loss = loss.item()

print(f'''\nModel saved at epoch {best_model_epoch} with train acc: {best_model_train_accuracy:.4f}%
                                                       valid acc: {best_model_valid_accuracy:.4f}%''')
# %%
from matplotlib import pyplot as plt

x1 = [i+1 for i in range(len(train_loss))]

# 첫 번째 subplot: 손실
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(x1, train_loss, 'b', label='Train Loss')
plt.plot(x1, valid_loss, 'r', label='Validation Loss')
plt.axvline(x=best_model_epoch, color='grey', linestyle='--', label='Best Model Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

# 두 번째 subplot: 정확도
plt.subplot(1, 2, 2)
plt.plot(x1, train_accuracy, 'g', label='Train Accuracy')
plt.plot(x1, valid_accuracy, 'orange', label='Validation Accuracy')
plt.axvline(x=best_model_epoch, color='grey', linestyle='--', label='Best Model Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

plt.show()

# %%
# Flatten the test_X data
X = []
for region in test_X.keys():
    for file in test_X[region].keys():
        # DataFrame에서 불필요한 열을 제거하고 값들을 추출
        data = test_X[region][file].drop(columns=['Unnamed: 0']).values
        normalized_data = scaler.transform(data)

        # 정규화된 데이터를 1차원 배열로 변환하여 리스트에 추가
        X.append(normalized_data.flatten())

# Prepare the test_y labels for the four categories
y_memberNo = []
y_jobType = []
y_houseType = []
y_houseArea = []

for region in test_y.keys():
    for file in test_y[region].keys():
        # Append the label for each category to its respective list
        y_memberNo.append(test_y[region][file].query("type == 'memberNo'")['value'].iloc[0])
        y_jobType.append(test_y[region][file].query("type == 'jobType'")['value'].iloc[0])
        y_houseType.append(test_y[region][file].query("type == 'houseType'")['value'].iloc[0])
        y_houseArea.append(test_y[region][file].query("type == 'houseArea'")['value'].iloc[0])

# Convert lists to numpy arrays|
y_memberNo = np.array(y_memberNo)
y_jobType = np.array(y_jobType)
y_houseType = np.array(y_houseType)
y_houseArea = np.array(y_houseArea)
# %%
def calculate_accuracy(predicted, labels):
    predicted = predicted.tolist()
    labels = labels.tolist()
    print(type(predicted))
    print(type(labels))
    correct = [0, 0, 0]
    count = []
    for i in range(1, 4):
        count.append(labels.count(i))
    for i, j in zip(predicted, labels):
        if i == j:
            correct[j-1] += 1
    print(f'''Label 1: 전체 {count[0]}개 중 {correct[0]}개 맞춤, 예측 성공률 {correct[0] * 100/count[0]:.2f}%
Label 2: 전체 {count[1]}개 중 {correct[1]}개 맞춤, 예측 성공률 {correct[1] * 100/count[1]:.2f}%
Label 3: 전체 {count[2]}개 중 {correct[2]}개 맞춤, 예측 성공률 {correct[2] * 100/count[2]:.2f}%''')

# %%
from sklearn.metrics import f1_score

X = torch.Tensor(X)
y = torch.Tensor(y_memberNo)

X, y = X.to(device), y.to(device)

model.load_state_dict(torch.load('memberNo_best_model.pth'))

# CUDA 사용 가능 여부 확인하고 모델을 해당 디바이스로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 평가 모드로 설정
model.eval()

output = model(X)
labels = y - 1

_, predicted = torch.max(output.data, 1)

total = labels.size(0)
correct = (predicted == labels).sum().item()

predicted = np.array(predicted.cpu())
labels = np.array(labels.cpu())
f1 = f1_score(predicted, labels, average='weighted')

print(f"test accuracy: {correct * 100/total:.4f}%, f1 score: {f1:.4f}")
# %%
