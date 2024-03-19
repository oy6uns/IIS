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
'''기존의 scaling'''
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

M = []
# Flatten the train_X data
X = []
region = '1.suncheon'
for file in X_raw[region].keys():
    # DataFrame에서 불필요한 열을 제거하고 값들을 추출
    data = X_raw[region][file].drop(columns=['Unnamed: 0']).values
    normalized_data = scaler.fit_transform(data)

    # 정규화된 데이터를 1차원 배열로 변환하여 리스트에 추가
    X.append(normalized_data.flatten())
    M.append(sum(normalized_data.flatten())/len(normalized_data.flatten()))

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
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

'''0. 3sigma MinMax / 1. Robust Scaler / 2. MinMax Scaler / 3. StandardScaler'''
scaler_num = 2

# 원본 구조를 유지하기 위한 새로운 딕셔너리
scaled_X_raw = {}

for region in X_raw.keys():
    # 현재 region에 대한 모든 데이터를 합치기 위한 빈 데이터 프레임
    all_data_region = pd.DataFrame()
    
    # 현재 region의 파일 데이터 합치기
    for file in X_raw[region].keys():
        # 불필요한 열 제거
        data = X_raw[region][file].drop(columns=['Unnamed: 0'])
        # 데이터 추가
        all_data_region = pd.concat([all_data_region, data], ignore_index=True)

    if scaler_num == 0:
        '''3sigma MinMax'''
        # 전체 데이터에 대한 평균과 표준편차 계산
        mean = all_data_region.mean().mean()
        std = all_data_region.stack().std()

        # 평균 ±3시그마 경계 계산
        lower_bound = mean - 3*std
        upper_bound = mean + 3*std

        # 전체 데이터에서 ±3시그마 범위 내의 최대값과 최소값 찾기
        bounded_data = all_data_region[(all_data_region >= lower_bound) & (all_data_region <= upper_bound)]
        min_val = bounded_data.min().min()
        max_val = bounded_data.max().max()

        # 스케일링 적용
        scaled_data_region = (all_data_region - mean) / (max_val - min_val)
    
    else:
        if scaler_num == 1:
            '''Robust'''
            # RobustScaler 적용
            scaler = RobustScaler()
        if scaler_num == 2:
            '''MinMax'''
            # MinMaxScaler 적용
            scaler = MinMaxScaler()
        if scaler_num == 3:
            '''Standard'''
            # StandardScaler 적용
            scaler = StandardScaler()
        scaled_data_region = scaler.fit_transform(all_data_region)
        
        # 스케일링된 데이터를 DataFrame으로 변환
        scaled_data_region = pd.DataFrame(scaled_data_region, columns=all_data_region.columns)

    # 스케일링된 데이터를 원래의 파일 구조에 맞춰 딕셔너리로 저장
    scaled_X_raw[region] = {}
    start_idx = 0
    for file in X_raw[region].keys():
        # 원본 파일과 같은 행의 수 계산
        end_idx = start_idx + len(X_raw[region][file].index)
        # 해당 부분을 추출하여 딕셔너리에 저장
        scaled_X_raw[region][file] = scaled_data_region.iloc[start_idx:end_idx].reset_index(drop=True)
        # 다음 파일을 위한 인덱스 업데이트
        start_idx = end_idx
# %%
import numpy as np
from sklearn.preprocessing import StandardScaler

# Flatten the train_X data
X = []
region = '1.suncheon'
for file in scaled_X_raw[region].keys():
    # DataFrame에서 불필요한 열을 제거하고 값들을 추출
    data = scaled_X_raw[region][file].values

    # 정규화된 데이터를 1차원 배열로 변환하여 리스트에 추가
    X.append(data.flatten())

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
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# 데이터를 훈련, 검증, 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, target_y, test_size=0.2, random_state=42, stratify=target_y)

# SVM 모델 초기화 및 학습
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# 테스트 세트에 대한 예측
y_pred = svm_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, digits=4))
make_confusion_matrix(y_pred, y_test)
# %%
