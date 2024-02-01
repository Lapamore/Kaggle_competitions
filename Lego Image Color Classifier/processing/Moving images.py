import os
import shutil

path_to_data = 'YOUR_PATH'

list_files = [file for file in os.listdir(path_to_data) if os.path.splitext(file)[-1] not in ('.jpg', '.jpeg', '.png')]

for folder_name in list_files:
    for image_name in os.listdir(path_to_data + folder_name):
        des_path = f"{path_to_data}{folder_name}/{image_name}"
        new_path = path_to_data + image_name
        shutil.move(des_path, new_path)

    if len(os.listdir(path_to_data + folder_name)) == 0:
        os.rmdir(path_to_data + folder_name)

print('All files have been moved to the path {}'.format(path_to_data))
