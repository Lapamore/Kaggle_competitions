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

# # Создание директории и перемещение файлов
# if not os.path.isdir('data/imgs'):
#     os.mkdir(path + 'imgs')

# new_path = 'data/imgs'

# # Создание CSV файла и запись меток
# with open('labels.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     field = ["name", "label"]
#     writer.writerow(field)
#     for index, folder in enumerate(os.listdir(path)):
#         for name_img in os.listdir(os.path.join(path, folder)):
#             name, _ = os.path.splitext(name_img)
#             writer.writerow([name, index])
#             shutil.move(os.path.join(path, folder, name_img), os.path.join(new_path, name_img))

# print(f'Все файлы были перемещены в {new_path}!')
