import os
from tqdm import tqdm
import csv
import shutil

path = 'YOUR_PATH'
classes = os.listdir(path)

# Переименование файлов
for folder in tqdm(classes):
    path_to_image = os.path.join(path, folder)
    for index, el in enumerate(os.listdir(path_to_image)):
        _, format_ = os.path.splitext(el)

        last_name = os.path.join(path, folder, el)
        new_name = os.path.join(path, folder, f'{folder}_{index+1}{format_}')
        os.rename(last_name, new_name)

print('Все файлы переименованы')
