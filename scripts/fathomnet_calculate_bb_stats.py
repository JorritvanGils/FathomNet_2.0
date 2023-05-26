import os
import numpy as np
from tqdm import tqdm
# all_labels = "/home/ubuntu/ml/fathomnet/datasets/fathomnet_yolo/train/labels"
labels_path = "/media/jorrit/Storage SSD/fathomnet/datasets/FN_23_yolo/train/labels"

all_label_sizes = np.zeros((1000,), dtype=int) #tensor 0's


for labels_txt in tqdm(os.listdir(labels_path)):
    f = os.path.join(labels_path, labels_txt)
    if os.path.isfile(f): # if file exists
        label_file = open(f, 'r')
        for line in label_file:
            line = line.split(" ")
            # class_value = line[0]
            # x_center = line[1]
            # y_center = line[2]
            width = float(line[3])
            height = float(line[4])
            
            # from graph distribution_bbox_sizes we find that the average size
            # of a bbox is 2073
            # image_width = 1920
            # image_height = 1080
            # actual dimentions image = 1920*1080 = 2073600 / 1000 
            # box_surface represents number of pixels covered by the bounding box (max = 2073)

            box_surface = int(width * height * 2073)   #1920*1080 is 2073600. Round down by 1000
            if box_surface > 999:
                box_surface = 999
            all_label_sizes[box_surface] = all_label_sizes[box_surface] + 1

# print(all_label_sizes)
# # 839 boxes cover 1 pixel
# # 2032 boxes cover 2 pixels, etc.

# in distribution.out Sean added all occurences from 100 - 2073 pixes together.
# therefore 7.750000000000000000e+02 is a very large value. 

np.savetxt('/media/jorrit/Storage SSD/fathomnet/export/bbox_size_distr_train.out_1000', all_label_sizes, delimiter=',')