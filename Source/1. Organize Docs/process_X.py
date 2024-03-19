# raw data를 1년 하루 24point의 시간 단위를 가지는 csv로 변환하여 저장합니다. 
import os
import re
from datetime import datetime, time

import pandas as pd
import numpy as np

region_array = ['1.suncheon', '2.mokpo', '3.yeosu', '4.gwangyang', '5.naju']
dates = pd.date_range(start='2021-09-01', end='2022-08-31', freq='D')

for region in region_array:
    folder_path = f'/Users/saint/Documents/Gist/Lab/2023-2024 Winter/RawData/Validation/01.원천데이터/VS_{region}'
    file_list = sorted(os.listdir(folder_path))

    if '.DS_Store'in file_list:
        file_list.remove('.DS_Store')

    # 폴더가 존재하지 않을 시에만 추가
    save_path = f'./Data/01.원천데이터/Validation/{region}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(file_list)
    for xls_file in file_list:
        file_path = os.path.join(folder_path, xls_file)
        # 2021-09-01부터 2022-08-31까지 8760 Point의 데이터만 추출
        print(xls_file)
        file_df = pd.read_csv(file_path)[-8760:]

        # 각 파일의 데이터를 담을 빈 데이터프레임을 제작
        df = pd.DataFrame()
        for x in range(24):
            df[x] = None
        df.columns = [time(hour=x) for x in range(24)]
        # 날짜 범위로 인덱스를 설정
        df = df.reindex(dates)

        # 실제 데이터에서 24 point씩 잘라서 df에 삽입
        for i, start in enumerate(range(0, 8760, 24)):
            df.loc[dates[i]] = file_df.iloc[start:start+24, 2].values
        
        df.to_csv(f'./Data/01.원천데이터/Validation/{region}/{xls_file}')
