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
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

'''0. 3sigma MinMax / 1. Robust Scaler / 2. MinMax Scaler / 3. StandardScaler'''
scaler_num = 3

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
import statistics

from sklearn.preprocessing import StandardScaler

# Flatten the train_X data
label_data = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
label_mean = {1: [0]*24, 2: [0]*24, 3: [0]*24}
label_std = {1: [0]*24, 2: [0]*24, 3: [0]*24}
label_count = [0, 0, 0]

region = '1.suncheon'
excluded_rows_sum = 0

dict = {key: [] for key in range(24)}

for file_X, file_y in zip(scaled_X_raw[region].keys(), y_raw[region].keys()):
    label = y_raw[region][file_y].query("type == 'memberNo'")['value'].iloc[0]
    data = scaled_X_raw[region][file_X]

    for t in dict.keys():
        dict[t].append(round(sum(data.iloc[:, t:t+1].values.flatten())/len(data.iloc[:, t:t+1].values.flatten()), 2))

    _df = label_data[label]
    label_data[label] = pd.concat([_df, data], ignore_index=True)

    label_count[label-1] += 1

for label in range(1, 4):
    label_mean[label] = label_data[label].mean().values
    label_std[label] = label_data[label].std().values

print(f'''라벨 별 평균 전력 데이터
      1: {label_mean[1]}
    2: {label_mean[2]}
    3: {label_mean[3]}''')

print(f'''라벨 별 평균 전력 데이터
      1: {label_std[1]}
    2: {label_std[2]}
    3: {label_std[3]}''')

print('라벨 별 데이터 수:', label_count)
# %%
import matplotlib.pyplot as plt

# 4x6 그리드에 subplot 배치하여 다시 그리기
fig, axes = plt.subplots(4, 6, figsize=(20, 15))  # 4x6 그리드에 subplot 배치
fig.tight_layout(pad=3.0)  # subplot 간 간격 조정

for i, (key, data) in enumerate(dict.items()):
    row, col = divmod(i, 6)  # 행과 열 위치 계산
    ax = axes[row, col]
    ax.hist(data, bins=20)  # 각 subplot에 히스토그램 그리기
    ax.set_title(f'Hour {key}')  # 각 subplot에 시간대 제목 설정

plt.show()
# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

hours = [f"{hour:02d}:00" for hour in range(24)]

# 정규화된 데이터를 이용하여 하나의 그래프에 모두 나타내기
plt.figure(figsize=(15, 5))

# 정규화된 데이터의 꺾은선 그래프
plt.plot(hours, label_mean[1], marker='o', color='red', label='Detached House ')
plt.plot(hours, label_mean[1]-label_std[1], marker='', color='red', label='Detached House ', alpha = 0.3)
plt.plot(hours, label_mean[1]+label_std[1], marker='', color='red', label='Detached House ', alpha = 0.3)
plt.plot(hours, label_mean[2], marker='o', color='dodgerblue', label='Apartment')
plt.plot(hours, label_mean[2]-label_std[2], marker='', color='dodgerblue', label='Apartment', alpha = 0.3)
plt.plot(hours, label_mean[2]+label_std[2], marker='', color='dodgerblue', label='Apartment', alpha = 0.3)
plt.plot(hours, label_mean[3], marker='o', color='forestgreen', label='Villa')
plt.plot(hours, label_mean[3]-label_std[3], marker='', color='forestgreen', label='Villa', alpha = 0.3)
plt.plot(hours, label_mean[3]+label_std[3], marker='', color='forestgreen', label='Villa', alpha = 0.3)

plt.fill_between(hours, label_mean[1]-label_std[1], label_mean[1]+label_std[1], facecolor='red', alpha = 0.1)
plt.fill_between(hours, label_mean[2]-label_std[2], label_mean[2]+label_std[2], facecolor='dodgerblue', alpha = 0.1)
plt.fill_between(hours, label_mean[3]-label_std[3], label_mean[3]+label_std[3], facecolor='forestgreen', alpha = 0.1)

# 그래프 제목과 축 레이블 설정
plt.title('Normalized Mean Power Usage Throughout the Day\n', fontsize = 20)
plt.xlabel('Time of Day', fontsize = 20)
plt.ylabel('Normalized Mean Power', fontsize = 20)
# plt.legend(fontsize = 20)
plt.grid(True)
plt.xticks(rotation=45, fontsize = 12)

plt.show()
# %%
