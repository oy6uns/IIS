import os

import pandas as pd
import pickle

# 각 지역명을 key로 가지는 딕셔너리 생성 및 초기화 
region_array = ['1.suncheon', '2.mokpo', '3.yeosu', '4.gwangyang', '5.naju']
train_X = {region: {} for region in region_array}
train_y = {region: {} for region in region_array}
test_X = {region: {} for region in region_array}
test_y = {region: {} for region in region_array}

dicts = [train_X, train_y, test_X, test_y]
folder_path_array = ['Data/01.원천데이터/Training', 'Data/02.라벨링데이터/Training', 'Data/01.원천데이터/Validation', 'Data/02.라벨링데이터/Validation']

for folder_path, dict in zip(folder_path_array, dicts):
    for region in region_array:
        inner_folder_path = os.path.join(folder_path, region)
        file_list = sorted(os.listdir(inner_folder_path))

        for file_name in file_list:
            file_path = os.path.join(inner_folder_path, file_name)
            df = pd.read_csv(file_path)

            file_key = file_name.split('.')[0]

            # dictionary에 dataframe 저장
            dict[region][file_key] = df

# 생성한 딕셔너리를 파일로 저장
base_save_path = 'Source/0. Dicts'
dicts_to_save = {'train_X.pkl': train_X, 'train_y.pkl': train_y, 'test_X.pkl': test_X, 'test_y.pkl': test_y}

for file_name, dict_to_save in dicts_to_save.items():
    pickle_file_path = os.path.join(base_save_path, file_name)
    
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(dict_to_save, file)

print("Dictionaries have been saved as pickle files.")



