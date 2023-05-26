import json

with open('C:/Users/jorri/fathomnet/fathomnet-out-of-sample-detection/object_detection/train.json', 'r') as f:
    data_train = json.load(f)
    
with open('C:/Users/jorri/fathomnet/fathomnet-out-of-sample-detection/object_detection/eval.json', 'r') as f:
    data_eval = json.load(f)

num_images_train = len(data_train['images'])
num_images_eval = len(data_eval['images'])

print("Number of images train:", num_images_train)
print("Number of images eval:", num_images_eval)
