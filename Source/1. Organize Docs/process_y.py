import os
import re
from datetime import datetime, time

import pandas as pd
import numpy as np

region_array = ['1.suncheon', '2.mokpo', '3.yeosu', '4.gwangyang', '5.naju']

for region in region_array:
    folder_path = f'/Users/saint/Documents/Gist/Lab/2023-2024 Winter/RawData/Validation/02.라벨링데이터/VL_{region}'
    file_list = sorted(os.listdir(folder_path))

    if '.DS_Store'in file_list:
        file_list.remove('.DS_Store')

    # 폴더가 존재하지 않을 시에만 추가
    save_path = f'./Data/02.라벨링데이터/Validation/{region}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(file_list)
    for xls_file in file_list:
        file_path = os.path.join(folder_path, xls_file)
        # 2021-09-01부터 2022-08-31까지 8760 Point의 데이터만 추출
        print(xls_file)
        file_df = pd.read_csv(file_path, encoding='ISO-8859-1')
        file_df.rename(columns={'QITM_EN': 'type', 'RSPNS_CN': 'value'}, inplace=True)
        df = file_df.iloc[0:7, 3:5]
        df.to_csv(f'./Data/02.라벨링데이터/Validation/{region}/{xls_file}', index=False)