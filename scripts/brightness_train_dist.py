import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


# calculate the average brightness value for each image in a folder 
folder_path = "/media/jorrit/Storage SSD/fathomnet/datasets/FN_23_yolo/train/images"  

# folder_path = "/media/jorrit/Storage SSD/fathomnet/datasets/sea_animals"
output_file = '/media/jorrit/Storage SSD/fathomnet/export/brightness_distribution.png'

threshold = 100 
brightness_values = []

# Loop through all the files in the folder
for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # print(filename)
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Calculate the average brightness value
        # Brightness value of 0     = dark
        # Brightness value of 255   = white
        avg_brightness = cv2.mean(img)[0]
        if avg_brightness <= 0.0:
            print(f"Anomaly detected in image '{filename}': average brightness is {avg_brightness}")
        elif avg_brightness >= 255.0:
            print(f"Anomaly detected in image '{filename}': average brightness is {avg_brightness}")
        brightness_values.append(avg_brightness)
# print(brightness_values)
# Anomaly detected in image 'e8a7a794-2b80-41aa-80d2-1378830f6f81.png': average brightness is 0.0
# Anomaly detected in image '2e20d41e-b9a6-4874-acfd-322aec84c19b.png': average brightness is 0.0

# Plot the results in a distribution graph
plt.hist(brightness_values, bins=50)
plt.xlabel("Brightness Value")
plt.ylabel("Number of Images")
plt.title("Image brightness Distribution")
# plt.show()
plt.savefig(output_file)

# Save list to a file in the desired directory
with open('/media/jorrit/Storage SSD/fathomnet/export/train_brightness_list.pkl', 'wb') as f:
    pickle.dump(brightness_values, f)