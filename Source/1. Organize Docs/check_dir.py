import os
import re
from datetime import datetime, time

import pandas as pd
import numpy as np


# 해당 폴더의 파일 수 출력
base_path = 'Data/01.원천데이터/Training'

def count_files(directory):
    for dirpath, dirnames, files in os.walk(directory):
        print(f"In directory {os.path.basename(dirpath)}: {len(files)} file(s)")

count_files(base_path)

# The function below compares the file lists from two directories and prints out the names of files
# that are only present in one of the directories.

# def print_unique_files(dir1_files, dir2_files):
#     # Convert lists to sets for efficient comparison
#     set_dir1 = set(dir1_files)
#     set_dir2 = set(dir2_files)
    
#     # Find files unique to dir1 (in dir1 but not in dir2)
#     unique_to_dir1 = set_dir1 - set_dir2
#     # Find files unique to dir2 (in dir2 but not in dir1)
#     unique_to_dir2 = set_dir2 - set_dir1
    
#     # Print out the unique file names
#     if unique_to_dir1:
#         print(f"Files only in the first directory: {unique_to_dir1}")
#     if unique_to_dir2:
#         print(f"Files only in the second directory: {unique_to_dir2}")

# # Assume these are the file lists in the respective directories
# for dirpath, dirnames, files in os.walk('/Users/saint/Documents/Gist/Lab/2023-2024 Winter/Training/01.원천데이터/TS_5.naju'):
#     files_in_dir_A = files
# for dirpath, dirnames, files in os.walk('/Users/saint/Documents/Gist/Lab/2023-2024 Winter/Training/02.라벨링데이터/TL_5.naju'):
#     files_in_dir_B = files

# # Call the function to print out unique file names
# print_unique_files(files_in_dir_A, files_in_dir_B)



