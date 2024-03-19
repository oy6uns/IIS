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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()

# Flatten the train_X data
X_origin = []
y_memberNo = []
y_jobType = []
y_houseType = []
y_houseArea = []

count = [0, 0, 0]

region = '1.suncheon'
# for file_X, file_y in zip(X_raw[region].keys(), y_raw[region].keys()):
#     label_index = y_raw[region][file_y].query("type == 'memberNo'")['value'].iloc[0]
    
#     if count == 15:
#         break
    
#     if count[label_index - 1] < 5:
#         # DataFrame에서 불필요한 열을 제거하고 값들을 추출
#         data = X_raw[region][file_X].drop(columns=['Unnamed: 0']).values
#         normalized_data = scaler.fit_transform(data)

#         # 정규화된 데이터를 1차원 배열로 변환하여 리스트에 추가
#         X_origin.append(normalized_data.flatten())

#         y_memberNo.append(y_raw[region][file_y].query("type == 'memberNo'")['value'].iloc[0])
#         y_jobType.append(y_raw[region][file_y].query("type == 'jobType'")['value'].iloc[0])
#         y_houseType.append(y_raw[region][file_y].query("type == 'houseType'")['value'].iloc[0])
#         y_houseArea.append(y_raw[region][file_y].query("type == 'houseArea'")['value'].iloc[0])

#         count[label_index - 1] += 1
#     else:
#         continue

for file_X, file_y in zip(X_raw[region].keys(), y_raw[region].keys()):    
    # DataFrame에서 불필요한 열을 제거하고 값들을 추출
    data = X_raw[region][file_X].drop(columns=['Unnamed: 0'])
    normalized_data = scaler.fit_transform(data)

    # 정규화된 데이터를 1차원 배열로 변환하여 리스트에 추가
    X_origin.append(normalized_data.flatten())

    y_memberNo.append(y_raw[region][file_y].query("type == 'memberNo'")['value'].iloc[0])
    y_jobType.append(y_raw[region][file_y].query("type == 'jobType'")['value'].iloc[0])
    y_houseType.append(y_raw[region][file_y].query("type == 'houseType'")['value'].iloc[0])
    y_houseArea.append(y_raw[region][file_y].query("type == 'houseArea'")['value'].iloc[0])

# %%
# import matplotlib.pyplot as plt

# LIMIT = 8760
# index = [i for i in range(0, LIMIT)]
# index_labels = [f'day{day} {hour}h' for day in range(1, 366) for hour in range(0, 24)]

# # 라벨 1에 대한 그래프
# plt.figure(figsize=(400,40))
# for X, y in zip(X_origin, y_memberNo):
#     if y == 1:
#         plt.plot(index, X[:LIMIT], color='red', linestyle='-', marker='.', alpha = 0.1)
# plt.title('Label 1')
# plt.xticks(ticks=range(0, LIMIT, 24), labels=index_labels[::24])
# plt.show()

# # 라벨 2에 대한 그래프
# plt.figure(figsize=(400,40))
# for X, y in zip(X_origin, y_memberNo):
#     if y == 2:
#         plt.plot(index, X[:LIMIT], color='blue', linestyle='-', marker='.', alpha = 0.1)
# plt.title('Label 2')
# plt.xticks(ticks=range(0, LIMIT, 24), labels=index_labels[::24])
# plt.show()

# # 라벨 3에 대한 그래프
# plt.figure(figsize=(400,40))
# for X, y in zip(X_origin, y_memberNo):
#     if y == 3:
#         plt.plot(index, X[:LIMIT], color='green', linestyle='-', marker='.', alpha = 0.1)
# plt.title('Label 3')
# plt.xticks(ticks=range(0, LIMIT, 24), labels=index_labels[::24])
# plt.show()

# # %%
# import matplotlib.pyplot as plt

# LIMIT = 8760
# index = [i for i in range(0, LIMIT)]
# index_labels = [f'day{day} {hour}h' for day in range(1, 366) for hour in range(0, 24)]

# # 라벨 1에 대한 그래프
# plt.figure(figsize=(20,5))
# for X, y in zip(X_origin, y_memberNo):
#     if y == 1:
#         plt.plot(index, X[:LIMIT], color='red', linestyle='-', marker='.', alpha = 0.04)
#     elif y == 2:
#         plt.plot(index, X[:LIMIT], color='blue', linestyle='-', marker='.', alpha = 0.04)
#     else:
#         plt.plot(index, X[:LIMIT], color='green', linestyle='-', marker='.', alpha = 0.04)
# plt.xticks(ticks=range(0, LIMIT, 24), labels=index_labels[::24])
# plt.show()
# %%
print(len(X_origin), len(X_origin[0]))
# %%
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
import numpy as np

# 가정: time_series_data는 1278개의 일차원 시계열 데이터를 포함하는 리스트입니다.
# time_series_data = [ts1, ts2, ..., ts1278], 여기서 ts1, ts2 등은 각각 8640개의 timepoint를 가진 일차원 리스트입니다.

# 시계열 데이터를 스케일링합니다 (옵션).
time_series_data_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_origin)

# TimeSeriesKMeans 인스턴스 생성 및 클러스터링 수행
# n_clusters는 생성할 클러스터의 수를 나타냅니다.
seed = 0  # 재현 가능한 결과를 위한 난수 시드
n_clusters = 5
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True, random_state=seed)
labels = model.fit_predict(time_series_data_scaled)

# 클러스터링 결과 시각화
plt.figure(figsize=(12, 8))
for yi in range(n_clusters):
    plt.subplot(n_clusters, 1, yi + 1)
    for xx in time_series_data_scaled[labels == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(model.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 8640)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("TimeSeriesKMeans Clustering")

plt.tight_layout()
plt.show()
# %%
