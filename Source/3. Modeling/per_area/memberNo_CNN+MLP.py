# %%
import os
import sys

import pandas as pd
import pickle

from Func.Calculate import division, calculate_accuracy
from Func.Graph import make_confusion_matrix

base_load_path = '/home/user/Documents/oy6uns/Source/0. Dicts'

pickle_files = [f'X.pkl', f'y.pkl']

loaded_dicts = {}

for pickle_file in pickle_files:
    pickle_file_path = os.path.join(base_load_path, pickle_file)
    
    # 저장된 딕셔너리를 load해준다. 
    with open(pickle_file_path, 'rb') as file:
        loaded_dicts[pickle_file.replace('.pkl', '')] = pickle.load(file)

# 'loaded_dicts'에서 키를 사용하여 각 딕셔너리를 가져온다
X_raw = loaded_dicts[pickle_files[0].replace('.pkl', '')]
y_raw = loaded_dicts[pickle_files[1].replace('.pkl', '')]
# %%
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Flatten the train_X data
X = []
region = '1.suncheon'
for file in X_raw[region].keys():
    # DataFrame에서 불필요한 열을 제거하고 값들을 추출
    data = X_raw[region][file].drop(columns=['Unnamed: 0']).values
    normalized_data = scaler.fit_transform(data)

    # 정규화된 데이터를 1차원 배열로 변환하여 리스트에 추가
    X.append(normalized_data.flatten())

print(len(X))

# Prepare the train_y labels for the four categories
y_memberNo = []
y_jobType = []
y_houseType = []
y_houseArea = []

for file in y_raw[region].keys():
    # Append the label for each category to its respective list
    y_memberNo.append(y_raw[region][file].query("type == 'memberNo'")['value'].iloc[0])
    y_jobType.append(y_raw[region][file].query("type == 'jobType'")['value'].iloc[0])
    y_houseType.append(y_raw[region][file].query("type == 'houseType'")['value'].iloc[0])
    y_houseArea.append(y_raw[region][file].query("type == 'houseArea'")['value'].iloc[0])

# Convert lists to numpy arrays
X = np.array(X)
y_memberNo = np.array(y_memberNo)
y_jobType = np.array(y_jobType)
y_houseType = np.array(y_houseType)
y_houseArea = np.array(y_houseArea)

target_y = y_memberNo
# %%
print(X.shape)
print(target_y.shape)
# %%
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from Func.CustomLoss import FocalLoss, SigmoidF1Loss, MultiClassF1Loss

random.seed(42)

# train, valid, test 나누기
X_train, X_test, y_train, y_test = train_test_split(torch.Tensor(X), torch.Tensor(target_y), test_size=0.2, stratify = target_y)
X_train, X_valid, y_train, y_valid = train_test_split(torch.Tensor(X_train), torch.Tensor(y_train), test_size=0.25, stratify = y_train)

train_y_count = [y_train.tolist().count(1), y_train.tolist().count(2), y_train.tolist().count(3)]
valid_y_count = [y_valid.tolist().count(1), y_valid.tolist().count(2), y_valid.tolist().count(3)]
test_y_count = [y_test.tolist().count(1), y_test.tolist().count(2), y_test.tolist().count(3)]

print(train_y_count)
print(valid_y_count)
print(test_y_count)

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
val_dataset = CustomDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TimeSeriesClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=4) 
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=20, kernel_size=2, stride=2)  

        self.linear_CNN = nn.Linear(7300, 2048)
        self.linear_MLP = nn.Linear(input_size, 2048)
        
        self.fc1 = nn.Linear(2048 + 2048, 1024)  
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # MLP
        x1 = self.dropout(self.relu(self.linear_MLP(x)))

        # Conv
        x2 = x.view(-1, 1, 8760)
        x2 = self.relu(self.conv1(x2))
        x2 = self.relu(self.conv2(x2))
        x2 = self.relu(self.conv3(x2))

        x2 = x2.view(x2.size(0), -1)
        x2 = self.dropout(self.relu(self.linear_CNN(x2)))

        x_combined = torch.cat((x1, x2), dim=1)
        
        x_combined = self.dropout(self.relu(self.fc1(x_combined)))
        x_combined = self.dropout(self.relu(self.fc2(x_combined)))
        x_combined = self.dropout(self.relu(self.fc3(x_combined)))
        out = self.fc4(x_combined)
        
        return out


# CUDA 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
    
model = TimeSeriesClassifier(input_size=X_train.shape[1], num_classes=torch.unique(y_train).shape[0])
criterion = nn.CrossEntropyLoss()

if region == '1.suncheon':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
elif region == '2.mokpo':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
elif region == '3.yeosu':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    criterion = FocalLoss(alpha=[1, 0.3, 1], gamma=3, device=device)
elif region == '4.gwangyang':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
elif region == '5.naju':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = FocalLoss(alpha=[1, 0.3, 0.6], gamma=3, device=device)

# 모델을 GPU로 이동
model = model.to(device)

# ExponentialLR 스케줄러 정의
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# %%
train_loss = []
train_accuracy = []
valid_loss = []
valid_accuracy = []
_epoch = 1500
patience = 15  # 연속으로 손실이 개선되지 않는 에포크 허용 횟수
counter = 0   # 개선되지 않는 에포크를 카운트하는 변수
prev_loss = float('inf')  # 최소 손실을 저장하는 변수
best_val_loss = float('inf')  # best model을 선별하기 위한 변수

# Train
for epoch in range(_epoch):
    correct = 0
    total = 0
    train_loss_epoch = []
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        output = model(data)
        loss = criterion(output, labels)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss_epoch.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_epoch_accuracy = 100 * correct / total
    train_loss.append(np.mean(train_loss_epoch))
    train_accuracy.append(train_epoch_accuracy)
    print(f'Epoch [{epoch+1}/{_epoch}], Loss: {loss.item():.4f}, Accuracy: {train_epoch_accuracy}%')

    # Validation
    model.eval()
    all_predicted = torch.tensor([], dtype=torch.long).to(device)  # 누적할 predicted를 위한 빈 텐서
    all_labels = torch.tensor([], dtype=torch.long).to(device)  # 누적할 labels를 위한 빈 텐서

    with torch.no_grad():
        correct = 0
        total = 0
        valid_loss_epoch = []
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            loss = criterion(output, labels)

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            valid_loss_epoch.append(loss.item())

            # predicted와 labels를 누적하여 저장
            all_predicted = torch.cat((all_predicted, predicted), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)


    valid_epoch_accuracy = 100 * correct / total
    valid_loss.append(np.mean(valid_loss_epoch))
    valid_accuracy.append(valid_epoch_accuracy)
    print(f'validation set Loss: {np.mean(valid_loss_epoch):.4f}, Accuracy: {valid_epoch_accuracy}%')

    if np.mean(valid_loss_epoch) < best_val_loss:
        best_val_loss = np.mean(valid_loss_epoch)
        # 모델 상태 저장
        torch.save(model.state_dict(), f'model_CNN/memberNo_{region}_best_model.pth')
        best_model_epoch = epoch+1
        best_model_train_accuracy = train_epoch_accuracy
        best_model_valid_accuracy = valid_epoch_accuracy
        calculate_accuracy(all_predicted, all_labels)
        all_predicted = np.array(all_predicted.cpu())
        all_labels = np.array(all_labels.cpu())
        f1 = f1_score(all_labels, all_predicted, average='weighted')
        print(f'''Model saved at epoch {epoch+1} with validation acc: {valid_epoch_accuracy}%, f1 score: {f1:.3f}''')

    # Early Stopping Check
    if np.mean(valid_loss_epoch) > prev_loss:
        counter += 1
        if counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    else:
        counter = 0

    prev_loss = np.mean(valid_loss_epoch)

calculate_accuracy(all_predicted, all_labels)
print(f'''Model saved at epoch {best_model_epoch} with train acc: {best_model_train_accuracy:.3f}%, valid acc: {best_model_valid_accuracy:.3f}%, f1 score: {f1:.3f}''')
# %%
make_confusion_matrix(all_predicted, all_labels)
# %%
import matplotlib.pyplot as plt
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

X = torch.Tensor(X_test)
y = torch.Tensor(y_test)

X, y = X.to(device), y.to(device)

model.load_state_dict(torch.load(f'model_CNN/memberNo_{region}_best_model.pth'))

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

make_confusion_matrix(predicted, labels)
calculate_accuracy(predicted, labels.long())

# F1 score 계산
predicted = np.array(predicted.cpu())
labels = np.array(labels.cpu())
f1 = f1_score(labels, predicted, average='weighted')

# AUC value 계산
probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
onehot_encoder = OneHotEncoder(sparse=False)
labels_onehot = onehot_encoder.fit_transform(labels.reshape(-1, 1))
auc_score = roc_auc_score(labels_onehot, probabilities, multi_class='ovo', average='weighted')

print(f"test accuracy: {correct * 100/total:.3f}%, f1 score: {f1:.3f}, auc score: {auc_score:.3f}")
# %%
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import seaborn as sns

# Adjust labels and predicted arrays
labels = labels + 1
predicted = predicted + 1

# Sort labels and correctness based on the labels
sorted_indices = np.argsort(labels)
labels_sorted = labels[sorted_indices]
correctness_sorted = (predicted[sorted_indices] == labels_sorted).astype(float)

# Settings for the heatmap
num_per_row = 50  # Number of predictions per row

# Padding length to make the total length a multiple of 'num_per_row'
padding_length = (-len(labels_sorted)) % num_per_row

# Pad the sorted arrays with np.nan at the end to match the desired shape
labels_padded = np.pad(labels_sorted, (0, padding_length), constant_values=np.nan)
correctness_padded = np.pad(correctness_sorted, (0, padding_length), constant_values=np.nan)

# Calculate the number of rows needed
num_rows = int(np.ceil(len(labels_padded) / num_per_row))

# Reshape the padded arrays
labels_grid = labels_padded.reshape((num_rows, num_per_row))
correctness_grid = correctness_padded.reshape((num_rows, num_per_row))

# Mask to avoid annotating nan values
mask = np.isnan(correctness_grid)

# Create the custom colormap
cmap = matplotlib.colors.ListedColormap(['red', 'green'])
cmap.set_bad('white')

# Plot the heatmap
plt.figure(figsize=(50, num_rows))
ax = sns.heatmap(correctness_grid, annot=labels_grid, cmap=cmap, cbar=False,
                 linewidths=.5, linecolor='grey', fmt='.0f', mask=mask,
            annot_kws={"size": 28})

# Increase the size of the annotations
for t in ax.texts: t.set_text(t.get_text() + " ")
ax.figure.axes[-1].yaxis.label.set_size(16)

plt.axis('off')  # Hide the axes
plt.show()
# %%
