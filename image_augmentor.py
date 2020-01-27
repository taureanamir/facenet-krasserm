import Augmentor
import os
import shutil

input_dir = "/mnt/drive/Amir/work/dataset/face_recognition_to_access_electronic_door_system/Augment_this_data/"

for dirname in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, dirname)
    p = Augmentor.Pipeline(path)

    p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
    # p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    p.flip_left_right(probability=0.8)
    p.zoom_random(probability=0.5, percentage_area=0.8)

    p.sample(750)


for dirname in sorted(os.listdir(input_dir)):
    src = os.path.join(input_dir, dirname, 'output')
    dst = os.path.join(input_dir, dirname)
    for file in os.listdir(src):
        # print(src)
        # print(dst)
        filepath = os.path.join(src,file)
        # if not os.path.exists(filepath):
        shutil.move(filepath, dst)

    shutil.rmtree(src)