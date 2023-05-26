import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import numpy as np

import requests
from PIL import Image
from io import BytesIO

# CONFIGS
# table display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# colors
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'

# get cwd
script_path = os.path.abspath(__file__)
fathomnet_path = os.path.abspath(os.path.join(script_path, '..', '..'))


# CONTENT
# 0.EXPLORE FILES
# 1.CATEGORY KEYS
# 2.TRAIN DATA
# 3.Train JSON
# 4.Evaluation DATA (JSON)
# 5.Export Files

# # 0.EXPLORE FILES
# # ADAPTED TO PYTHON CODE BY J.
# print('0.EXPLORE FILES')
# files = os.listdir('../FATHOMNET')
# for file in files:
#     print(file)


# 1.CATEGORY KEYS
print('1.CATEGORY KEYS')
df_catkey = pd.read_csv(os.path.join(fathomnet_path, 'category_key.csv'))
category_number = 160
row = df_catkey.loc[df_catkey['id'] == category_number]
category_name = row.iloc[0]['name']
print('most frequent category name = ', category_name)

# plt.figure(figsize=(8,3))
# df_catkey.supercat.value_counts().plot(kind='bar', color=default_color_1)
# plt.title('supercat - Distribution in Category Key Table')
# plt.grid()
# plt.show()

print(df_catkey.supercat.value_counts())

# 2.TRAIN DATA
print('2.TRAIN DATA')
# load multilabel_classification train data
df_train = pd.read_csv(os.path.join(fathomnet_path, 'multilabel_classification', 'train.csv'))

# display first few rows and statistics
print('multilabel_classification train.csv:\n', df_train.head(5))
print('describe:\n', df_train.describe())

# create graph label frequency occurences
def count_labels(i_list_as_str):
    return len(json.loads(i_list_as_str))
df_train['label_count'] = df_train.categories.apply(count_labels)

# plt.figure(figsize=(7,4))
# df_train.label_count.value_counts().plot(kind='bar', color=default_color_1)
# plt.title('Number of train images against label frequency')
# plt.grid()
# plt.show()

print('images with 10 labels:\n', df_train[df_train.label_count==10])


# 3.Train JSON
# Read the file as a string (read) so json.loads needed for dict

# directly into python dict (desired format)
with open(os.path.join(fathomnet_path, 'object_detection', 'train.json')) as f:
    dict_train = json.load(f)
    # json_data = f.read() # acces dict via more readable but less usable f.read 
# dict_train = json.loads(json_data) # json.loads needed to use

print(dict_train.keys())
print(dict_train['info'])
print(dict_train['licenses'])

df_train_images = pd.json_normalize(dict_train['images'])
df_train_images.drop(['license'], axis=1, inplace=True)
print(df_train_images.info())
print(df_train_images.head(5))

# # plot_width_and_height_of_images_train, does not work! 
# sns.jointplot(df_train_images, x='width', y='height',
#               color=default_color_1,
#               alpha=0.25)
# plt.grid()
# plt.show()

df_train_annots = pd.json_normalize(dict_train['annotations'])
print(df_train_annots.head(5))
df_train_annots.drop(['segmentation', 'iscrowd'], axis=1, inplace=True)
print(df_train_annots.info())
print('len(df_train_annots)', len(df_train_annots), 'len(df_train_images)', len(df_train_images), 
      'so on average an image has', len(df_train_annots)/len(df_train_images), 'annotations')

# # plot distribution of area
# plt.figure(figsize=(7,3))
# df_train_annots.area.plot(kind='hist', bins=100, color=default_color_1)
# plt.title('area')
# plt.grid()
# # plt.xlim(0, 1) # set the limits of the x-axis
# plt.show()

# print(df_train_annots[df_train_annots.area==0])

# # plot distribution of area (ignore one row having area=0)
# plt.figure(figsize=(7,3))
# np.log10(df_train_annots.area[df_train_annots.area!=0]).plot(kind='hist', 
#                                                              bins=100,
#                                                              color=default_color_1)
# plt.title('area distribution annotations - log_10 scale')
# plt.grid()
# plt.show()

# aux function (remove suffix ".png" from file name) of the train.json key 'images' (df_train_images) 
def drop_suffix(i_str):
    length = len(i_str)
    return i_str[0:(length-4)]

# add attribute['id_file'] with the image id
# add filename-based id to image table in order to be able to join with df_train
df_train_images['id_file'] = df_train_images.file_name.apply(drop_suffix)
# id_file column succesfully added. 
print(df_train_images.info())

# UNCOMMENT CODE OF 2.TRAIN DATA, FOR THE MERGE
# JOIN TRAINING TABLES:
# df_train -> originally train.csv (image id's + categories + count)
# df_train_images -> train.json ['images']
df_train_with_labels = df_train_images.merge(right=df_train, how='left',
                                             left_on='id_file', right_on='id')
df_train_with_labels.drop(['id_y'], axis=1, inplace=True)
# rename id column
df_train_with_labels.rename(columns={'id_x':'image_id'}, inplace=True)
# show preview of complete table
print(df_train_with_labels.head(5))
print(df_train_with_labels.info())

# EXAMPLE TO VIEW IMG AND GET ANNOTATIONS
# get URL and look at image
# my_row = 0
# or search based on an image_id
image_id = '3f735021-f5de-4168-b139-74bf2859d12a'
my_row = df_train_with_labels.loc[df_train_with_labels['id_file'] == image_id].index[0]
my_url = df_train_with_labels.coco_url[my_row]
response = requests.get(my_url)
my_img_train = Image.open(BytesIO(response.content))
# print(my_img_train.show())

# get category_ids for the image
my_annots = df_train_with_labels.categories[my_row]
my_annots = json.loads(my_annots)

# get category and annotation info
# UNCOMMENT CODE OF 1.CATEGORY KEYS, FOR THE MERGE
# create table with (category) name and (category) supercat(egory) 
df_img_info_cat = df_catkey[df_catkey.id.isin(my_annots)]
# print(df_img_info_cat)
# create table with annotation information 
df_img_ann_info = df_train_annots[df_train_annots.image_id == df_train_with_labels.image_id[my_row]]
# print(df_img_ann_info)
# join tables df_img_ann_info and df_img_ann_info_detail:
df_img_ann_info = df_img_info_cat.merge(right=df_img_ann_info, how='left',
                                             left_on='id', right_on='category_id')
df_img_ann_info.drop(['category_id'], axis=1, inplace=True)
df_img_ann_info = df_img_ann_info.rename(columns={'id_x': 'category_id', 'id_y': 'annotation_id'})
print(df_img_ann_info)


# 4.Evaluation DATA (JSON)
print('4.Evaluation DATA (JSON)')
with open(os.path.join(fathomnet_path, 'object_detection', 'eval.json')) as f:
    dict_eval = json.load(f)

print(dict_eval['info'])
print(dict_eval['licenses'])

# convert image section to data frame
df_eval_images = pd.json_normalize(dict_eval['images'])
df_eval_images.drop(['license'], axis=1, inplace=True)
# again add file name attribute based id
df_eval_images['id_file'] = df_eval_images.file_name.apply(drop_suffix)
print(df_eval_images.head(5))
print(df_eval_images.info())
print('len(df_train_images) = ', len(df_train_images),'len(df_eval_images) = ', len(df_eval_images))

# below does not work
# sns.jointplot(df_eval_images, x='width', y='height',
#               color=default_color_1,
#               alpha=0.25)
# plt.grid()
# plt.show()

my_row = 9
my_url = df_eval_images.coco_url[my_row]
my_url
response = requests.get(my_url)
my_img_eval = Image.open(BytesIO(response.content))
# my_img_eval.show()


# 5.Export Files
print('5.Export Files')
df_train_annots.to_csv(os.path.join(fathomnet_path, 'export', 'train_annotations.csv'), index=False)
df_train_with_labels.to_csv(os.path.join(fathomnet_path, 'export', 'train_with_labels.csv'), index=False)
df_img_ann_info.to_csv(os.path.join(fathomnet_path, 'export', 'img_ann_info.csv'), index=False)
df_eval_images.to_csv(os.path.join(fathomnet_path, 'export', 'eval_images.csv'), index=False)

# sample submission
df_sub = pd.read_csv(os.path.join(fathomnet_path, 'sample_submission.csv'))
print(df_sub.head(5))
print(df_sub.info())