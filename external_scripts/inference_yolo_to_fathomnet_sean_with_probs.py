import os
import csv

import fathomnet_calculate_prob_of_bb


# Change paths here
# exp_dir = '/media/jorrit/Storage SSD/fathomnet/yolov5/runs/detect/exp7'
exp_dir = '/home/ubuntu/ml/fathomnet/yolov5/runs/detect/exp/labels'
csv_name = '/home/ubuntu/ml/fathomnet/sample_submission.csv'

submission_lines = []
for filename in os.listdir(exp_dir):

    submission_line = []
    name = os.path.splitext(filename)[0] # -.png
    submission_line.append(name)
    
    # We want ordered pairs of (class, cumulative_prob) organized by highest prob
    seen_classes = {}
    with open(os.path.join(exp_dir, filename), "r") as label_file:
        for line in label_file:
            yolo_pred = line.split()

            # Check if this class was seen before
            if yolo_pred[0] not in seen_classes.keys():
                seen_classes[yolo_pred[0]] = yolo_pred[5]
            else:
                seen_classes[yolo_pred[0]] = seen_classes[yolo_pred[0]] + yolo_pred[5]

        # For all lines in a detection file we now have a dict of {class_num: cumulative_prob}
    sorted_classes_by_prob = sorted(seen_classes.items(), key=lambda x:x[1])
    ordered_classes = ""
    for pair in sorted_classes_by_prob:
        ordered_classes = ordered_classes + str(pair[0]) + " "
    ordered_classes = ordered_classes[:-1]
    submission_line.append(ordered_classes)

    labels = open(os.path.join(exp_dir, filename), "r")
    probs_of_bbs = fathomnet_calculate_prob_of_bb.probability_of_sample(labels)

    submission_line.append(sum(probs_of_bbs)/len(probs_of_bbs))
    submission_lines.append(submission_line)

with open(csv_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'categories', 'osd'])
    writer.writerows(submission_lines)

        














# # Directory path
# directory = "/media/jorrit/Storage SSD/fathomnet/yolov5/runs/detect/exp7"

# # Helper function to parse the .txt files
# def parse_txt_file(file_path):
#     with open(file_path, "r") as file:
#         lines = file.readlines()
#         results = []
#         for line in lines:
#             parts = line.strip().split(" ")
#             if len(parts) == 5:
#                 id = parts[0]
#                 categories = " ".join(parts[1:])
#                 result = f"{id},{categories},"
#                 results.append(result)
#         return results

# # Iterate through the directory and its subdirectories
# for dirpath, dirnames, filenames in os.walk(directory):
#     for file in filenames:
#         if file.endswith(".txt"):
#             file_path = os.path.join(dirpath, file)
#             results = parse_txt_file(file_path)
#             for result in results:
#                 print(result)
