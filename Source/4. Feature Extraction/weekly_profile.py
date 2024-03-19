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
scaler_num = 1

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
    
    elif scaler_num == 1:
        data_1d = all_data_region.values.flatten()
        # 전체 데이터에 대한 25%, 50%, 75% 사분위수 계산
        q25, q50, q75 = np.percentile(data_1d, [25, 50, 75])

        # Robust Scaling 적용
        scaled_data_region = (all_data_region - q50) / (q75 - q25)
    else:
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
print(q25, q50, q75)
# %%
import numpy as np
import statistics

from sklearn.preprocessing import StandardScaler

# Flatten the train_X data
label_count = [0, 0, 0]

region = '1.suncheon'
excluded_rows_sum = 0

label_value = {key: [] for key in range(1, 4)}

for file_X, file_y in zip(scaled_X_raw[region].keys(), y_raw[region].keys()):
    label = y_raw[region][file_y].query("type == 'memberNo'")['value'].iloc[0]
    data = scaled_X_raw[region][file_X].iloc[:-1]

    data = data.values.flatten()
    data = pd.DataFrame(data.reshape(52, 168))

    data = data.mean().values

    label_value[label].append(data)

    label_count[label-1] += 1

print('라벨 별 데이터 수:', label_count)
# %%

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

# Define the days of the week
days = ["Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue"]

# Create an array using list comprehension that combines the days with hours from 00 to 23
hours_in_week = [f"{day} {hour:02d}h" for day in days for hour in range(24)]

# 정규화된 데이터를 이용하여 하나의 그래프에 모두 나타내기
plt.figure(figsize=(15, 4))

# 정규화된 데이터의 꺾은선 그래프
for list in label_value[1]:
    plt.plot(hours_in_week, list, marker='', color='red', label='Detached House ', alpha = 0.1)
# for list in label_value[2]:
#     plt.plot(hours_in_week, list, marker='', color='dodgerblue', label='Apartment', alpha = 0.1)
# for list in label_value[3]:
#     plt.plot(hours_in_week, list, marker='', color='forestgreen', label='Villa', alpha = 0.1)

# 그래프 제목과 축 레이블 설정
plt.title('Normalized Mean Power Usage Throughout the Day\n', fontsize = 20)
plt.xlabel('Time of Day', fontsize = 20)
plt.ylabel('Normalized Mean Power', fontsize = 20)
# plt.legend(fontsize = 20)
plt.grid(True)
plt.xticks(rotation=45, fontsize = 12)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))


plt.show()
# %%
