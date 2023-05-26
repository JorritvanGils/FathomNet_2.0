import os
import numpy as np
from tqdm import tqdm
all_labels = "/home/ubuntu/ml/fathomnet/datasets/fathomnet_yolo/train/labels"

all_label_sizes = np.zeros((1000,), dtype=int)

for label_file_name in tqdm(os.listdir(all_labels)):
    f = os.path.join(all_labels, label_file_name)
    # checking if it is a file
    if os.path.isfile(f):
        label_file = open(f, 'r')
        for line in label_file:
            line = line.split(" ")
            box_surface = int(float(line[3]) * float(line[4]) * 2073)   #1920*1080 is 2073600. Round down by 1000
            if box_surface > 999:
                box_surface = 999
            all_label_sizes[box_surface] = all_label_sizes[box_surface] + 1

np.savetxt('distribution.out', all_label_sizes, delimiter=',')