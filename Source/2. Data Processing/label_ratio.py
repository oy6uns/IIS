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
for key in train_X.keys():
    train_X[key].update(test_X[key])
    train_y[key].update(test_y[key])  
# %%
import matplotlib.pyplot as plt

labels = [1, 2, 3]

label_size = 20
autopct_size = 24

fontdict_labels = {'fontsize': label_size}
fontdict_autopct = {'fontsize': autopct_size}

memberNo_total = []
jobType_total = []
houseArea_total = []
houseType_total = []

# train_y 예시 데이터를 여기에 넣으세요.
# 예: train_y = {'group1': {'memberNo': [1, 2, 3], 'jobType': [1, 2, 3], ...}}

for key in train_y.keys():
    memberNo = []
    jobType = []
    houseArea = []
    houseType = []
    for value in train_y[key].values():
        memberNo.append(value.iloc[0].value)
        jobType.append(value.iloc[1].value)
        houseArea.append(value.iloc[2].value)
        houseType.append(value.iloc[3].value)

    memberNo_total.extend(memberNo)
    jobType_total.extend(jobType)
    houseArea_total.extend(houseArea)
    houseType_total.extend(houseType)

    memberNo_count = [memberNo.count(1), memberNo.count(2), memberNo.count(3)]
    jobType_count = [jobType.count(1), jobType.count(2), jobType.count(3)]
    houseArea_count = [houseArea.count(1), houseArea.count(2), houseArea.count(3)]
    houseType_count = [houseType.count(1), houseType.count(2), houseType.count(3)]

    fig, axs = plt.subplots(4, 1, figsize=(5, 20))

    colors = ['#ff9999','#66b3ff','#99ff99']

    # 각 라벨의 개수를 표시하는 함수 정의
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
        return my_autopct

    axs[0].pie(memberNo_count, labels=labels, textprops=fontdict_labels, colors=colors, autopct=make_autopct(memberNo_count), wedgeprops=dict(width=0.7))
    axs[0].set_title('memberNo', fontdict=fontdict_labels)

    axs[1].pie(jobType_count, labels=labels, textprops=fontdict_labels, colors=colors, autopct=make_autopct(jobType_count), wedgeprops=dict(width=0.7))
    axs[1].set_title('jobType', fontdict=fontdict_labels)

    axs[2].pie(houseArea_count, labels=labels, textprops=fontdict_labels, colors=colors, autopct=make_autopct(houseArea_count), wedgeprops=dict(width=0.7))
    axs[2].set_title('houseArea', fontdict=fontdict_labels)

    axs[3].pie(houseType_count, labels=labels, textprops=fontdict_labels, colors=colors, autopct=make_autopct(houseType_count), wedgeprops=dict(width=0.7))
    axs[3].set_title('houseType', fontdict=fontdict_labels)

    plt.tight_layout()
    plt.show()

# %%
memberNo_count = [memberNo_total.count(1), memberNo_total.count(2), memberNo_total.count(3)]
jobType_count = [jobType_total.count(1), jobType_total.count(2), jobType_total.count(3)]
houseArea_count = [houseArea_total.count(1), houseArea_total.count(2), houseArea_total.count(3)]
houseType_count = [houseType_total.count(1), houseType_total.count(2), houseType_total.count(3)]
fig, axs = plt.subplots(4, 1, figsize=(5, 20))

colors = ['#ff9999','#66b3ff','#99ff99']

axs[0].pie(memberNo_count, labels = labels, textprops=fontdict_labels, colors = colors, autopct = '%1.1f%%', wedgeprops=dict(width=0.7))
axs[0].set_title('memberNo', fontdict=fontdict_labels)

axs[1].pie(jobType_count, labels = labels, textprops=fontdict_labels, colors = colors, autopct = '%1.1f%%', wedgeprops=dict(width=0.7))
axs[1].set_title('jobType', fontdict=fontdict_labels)

axs[2].pie(houseArea_count, labels = labels, textprops=fontdict_labels, colors = colors, autopct = '%1.1f%%', wedgeprops=dict(width=0.7))
axs[2].set_title('houseArea', fontdict=fontdict_labels)

axs[3].pie(houseType_count, labels = labels, textprops=fontdict_labels, colors = colors, autopct = '%1.1f%%', wedgeprops=dict(width=0.7))
axs[3].set_title('houseType', fontdict=fontdict_labels)

plt.tight_layout()
plt.show()
# %%
from collections import Counter

# 데이터 순회 및 카운트
for key in train_y.keys():
    combinations_counter = []
    for value in train_y[key].values():
        combinations_counter.append(value.iloc[0:4]['value'].values)

    # 각 조합의 빈도수 계산
    counted_combinations = Counter(tuple(comb) for comb in combinations_counter)

    # 가장 빈번한 조합 10개를 내림차순으로 추출
    top_10_combinations = counted_combinations.most_common(10)

    # 결과 출력
    for i, (comb, count) in enumerate(top_10_combinations, 1):
        print(f"{i}. 순서쌍 {comb} : {count}개")
    print('\n')

# %%
print(train_y['1.suncheon'])
# %%
for key in train_y.keys():
    print(key)
# %%
