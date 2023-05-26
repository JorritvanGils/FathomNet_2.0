import csv
import os
import pandas as pd
import pickle

script_path = os.path.abspath(__file__)
fathomnet_path = os.path.abspath(os.path.join(script_path, '..', '..'))

# initialize an empty dictionary to store the class IDs and names
class_dict = {}

# read the category file and populate the class dictionary
with open(os.path.join(fathomnet_path, 'category_key.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_id = int(row['id']) - 1  # subtract 1 to match the IDs used in the original code
        class_name = row['name']
        class_dict[class_id] = class_name

# # print the class dictionary to verify the mapping
# print(class_dict)

# # saving as dict
# with open('class_dict.pkl', 'wb') as f:
#     pickle.dump(class_dict, f)

print("class_dict:")
for key, value in class_dict.items():
    print(f"  {key}: {value}")



