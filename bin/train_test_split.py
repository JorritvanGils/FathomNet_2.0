import os
import shutil
import tqdm as tq

# Set the path to the train folder
train_folder = "/media/jorrit/Storage SSD/fathomnet/datasets/FN_23_yolo/train"

# Set the path to the valid folder
valid_folder = "/media/jorrit/Storage SSD/fathomnet/datasets/FN_23_yolo/valid"

# Set the number of images and labels to move
num_files_to_move = 900

# Set the paths to the images and labels subfolders
images_folder = os.path.join(train_folder, "images")
labels_folder = os.path.join(train_folder, "labels")

# Get a list of the image and label files
image_files = sorted(os.listdir(images_folder))
label_files = sorted(os.listdir(labels_folder))

# Make sure there are enough files to move
if len(image_files) < num_files_to_move or len(label_files) < num_files_to_move:
    print("Not enough files to move")
    exit()

# Move the first num_files_to_move images to the valid folder
for i in tq.tqdm(range(num_files_to_move)):
    image_file = image_files[i]
    source_path = os.path.join(images_folder, image_file)
    dest_path = os.path.join(valid_folder, "images", image_file)
    shutil.move(source_path, dest_path)

# Move the first num_files_to_move labels to the valid folder
for i in tq.tqdm(range(num_files_to_move)):
    label_file = label_files[i]
    source_path = os.path.join(labels_folder, label_file)
    dest_path = os.path.join(valid_folder, "labels", label_file)
    shutil.move(source_path, dest_path)

print(f"{num_files_to_move} files moved from train to valid folder")

# Delete the first num_files_to_move images and labels from the train folder
for i in tq.tqdm(range(num_files_to_move)):
    image_file = image_files[i]
    label_file = label_files[i]
    image_source_path = os.path.join(images_folder, image_file)
    label_source_path = os.path.join(labels_folder, label_file)
    os.remove(image_source_path)
    os.remove(label_source_path)

print(f"{num_files_to_move} files deleted from train folder")



# go to the train folder (/media/jorrit/Storage SSD/fathomnet/datasets/FN_23_yolo/train)
# from subfolder images move first 900 images to /media/jorrit/Storage SSD/fathomnet/datasets/FN_23_yolo/valid/images
# from subfolder labels move first 900 labels to /media/jorrit/Storage SSD/fathomnet/datasets/FN_23_yolo/valid/labels