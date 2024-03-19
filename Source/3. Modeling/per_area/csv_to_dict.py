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

for key in train_X.keys():
    train_X[key].update(test_X[key])
    train_y[key].update(test_y[key])    

X = train_X
y = train_y

# 생성한 딕셔너리를 파일로 저장
base_save_path = f'/home/user/Documents/oy6uns/Source/0. Dicts'
dicts_to_save = {
        f'X.pkl': X, 
        f'y.pkl': y
}

for file_name, dict_to_save in dicts_to_save.items():
    pickle_file_path = os.path.join(base_save_path, file_name)
    
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(dict_to_save, file)

print(f"Dictionaries have been saved as pickle files.")
# %%
from itertools import islice

# 딕셔너리 split을 하기 위한 split_dict 함수 구현
def split_dict(original_dict, n):
    # 처음 n개의 항목을 추출
    first_n = dict(islice(original_dict.items(), n))
    
    # 나머지 항목을 추출
    rest = dict(islice(original_dict.items(), n, None))

    return first_n, rest
# %%
# 지역, 라벨 종류에 따라 다르게 데이터셋 생성
def custom_dataset(target_y, index):
    for region in X.keys():
        num_target = []
        dict_label_1_X = {}
        dict_label_2_X = {}
        dict_label_3_X = {}
        dict_label_1_y = {}
        dict_label_2_y = {}
        dict_label_3_y = {}
        for file_y in y[region].values():
            num_target.append(file_y.query(f"type == '{target_y}'")['value'].iloc[0])
        # target_y의 라벨 값별 개수를 저장
        target_y_count = [num_target.count(1), num_target.count(2), num_target.count(3)]

        print(target_y_count)

        for id, file_X, file_y in zip(train_X[region].keys(), train_X[region].values(), train_y[region].values()):
            label = file_y.query(f"type == '{target_y}'")['value'].iloc[0]
            # y 데이터프레임 중 target_y와 covid관련 feature만 추출
            print(file_y)
            selected_file_y = file_y.iloc[[index, 5, 6]]
            selected_file_y = selected_file_y.reset_index(drop=True)
            # target_y의 라벨 값별로 별도의 딕셔너리에 추가
            if label == 1:
                dict_label_1_X[id] = file_X
                dict_label_1_y[id] = selected_file_y
            elif label == 2:
                dict_label_2_X[id] = file_X   
                dict_label_2_y[id] = selected_file_y
            else:
                dict_label_3_X[id] = file_X
                dict_label_3_y[id] = selected_file_y
        
        # 각 라벨별로 분류한 딕셔너리에 대해 80:20의 비율로 분리
        dic_label_1_test_X, dict_label_1_train_X = split_dict(dict_label_1_X, target_y_count[0]//5)
        dic_label_2_test_X, dict_label_2_train_X = split_dict(dict_label_2_X, target_y_count[1]//5)
        dic_label_3_test_X, dict_label_3_train_X = split_dict(dict_label_3_X, target_y_count[2]//5)

        dic_label_1_test_y, dict_label_1_train_y = split_dict(dict_label_1_y, target_y_count[0]//5)
        dic_label_2_test_y, dict_label_2_train_y = split_dict(dict_label_2_y, target_y_count[1]//5)
        dic_label_3_test_y, dict_label_3_train_y= split_dict(dict_label_3_y, target_y_count[2]//5)

        # 1, 2, 3 라벨의 비율을 균등하게 다시 train, test 딕셔너리를 업데이트
        dict_train_X = {**dict_label_1_train_X, **dict_label_2_train_X, **dict_label_3_train_X}
        dict_test_X = {**dic_label_1_test_X, **dic_label_2_test_X, **dic_label_3_test_X}

        dict_train_y = {**dict_label_1_train_y, **dict_label_2_train_y, **dict_label_3_train_y}
        dict_test_y = {**dic_label_1_test_y, **dic_label_2_test_y, **dic_label_3_test_y}

        print(dict_test_y)

        print(len(dict_train_X), len(dict_test_X), len(dict_train_y), len(dict_test_y))
        
        # 생성한 딕셔너리를 파일로 저장
        base_save_path = f'/home/user/Documents/oy6uns/Source/0. Dicts/Region_base/{region}'
        dicts_to_save = {
                f'train_X_{region}_{target_y}.pkl': dict_train_X, 
                f'train_y_{region}_{target_y}.pkl': dict_train_y, 
                f'test_X_{region}_{target_y}.pkl': dict_test_X, 
                f'test_y_{region}_{target_y}.pkl': dict_test_y
        }
        
        for file_name, dict_to_save in dicts_to_save.items():
            pickle_file_path = os.path.join(base_save_path, file_name)
            
            with open(pickle_file_path, 'wb') as file:
                pickle.dump(dict_to_save, file)

        print(f"{region}_{target_y}_Dictionaries have been saved as pickle files.")
# %%
custom_dataset('memberNo', 0)
# %%
custom_dataset('jobType', 1)
# %%
custom_dataset('houseType', 2)
# %%
custom_dataset('houseArea', 3)
# %%
