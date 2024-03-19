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
3
# 'loaded_dicts'에 각 딕셔너리를 저장해준다. 
train_X = loaded_dicts['train_X']
train_y = loaded_dicts['train_y']
test_X = loaded_dicts['test_X']
test_y = loaded_dicts['test_y']
# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# %%
for key in test_y.keys():
    memberNo = []
    jobType = []
    houseArea = []
    houseType = []
    for value in train_y[key].values():
        memberNo.append(value.iloc[0].value)
        jobType.append(value.iloc[1].value)
        houseArea.append(value.iloc[2].value)
        houseType.append(value.iloc[3].value)
    for value in test_y[key].values():
        memberNo.append(value.iloc[0].value)
        jobType.append(value.iloc[1].value)
        houseArea.append(value.iloc[2].value)
        houseType.append(value.iloc[3].value)
    
    cm1 = confusion_matrix(memberNo, jobType)
    cm2 = confusion_matrix(memberNo, houseArea)
    cm3 = confusion_matrix(memberNo, houseType)
    cm4 = confusion_matrix(jobType, houseArea)
    cm5 = confusion_matrix(jobType, houseType)
    cm6 = confusion_matrix(houseArea, houseType)

    # 히트맵으로 시각화
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    cm_list = [cm1, cm2, cm3, cm4, cm5, cm6]
    titles = ['memberNo-jobType', 'memberNo-houseArea', 'memberNo-houseType', 
            'jobType-houseArea', 'jobType-houseType', 'houseArea-houseType']
    colors = ['Blues', 'Greens']

    for i, cm in enumerate(cm_list):
        ax = axes[i // 2, i % 2]
        cm_percentage = cm / cm.sum() * 100
        sns.heatmap(cm_percentage, annot=True, annot_kws={"size": 20}, cmap=colors[i % 2], ax=ax, )
        ax.set_title(titles[i], fontsize = 20)
        ax.set_xlabel(titles[i].split('-')[0], fontsize = 20)
        ax.set_ylabel(titles[i].split('-')[1], fontsize = 20)
        ax.tick_params(axis='x', labelsize=16) 
        ax.tick_params(axis='y', labelsize=16) 

    plt.suptitle(f'Relations between labels in {key}\n', fontsize=24)

    plt.tight_layout()
    plt.show()


# %%

# %%
