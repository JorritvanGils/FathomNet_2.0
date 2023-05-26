import os
import csv
import cv2

import fathomnet_calculate_prob_of_bb, brightness_value_prob

# Change paths here, Choose the folder with the eval images
exp_dir = '/home/ubuntu/ml/fathomnet/yolov5/runs/detect/exp/labels'
csv_name = '/home/ubuntu/ml/fathomnet/sample_submission.csv'
img_test_dir = '/home/ubuntu/ml/fathomnet/datasets/fathomnet_yolo/valid/images'


submission_lines = []
for filename in os.listdir(exp_dir):
    # print(filename)
    submission_line = []
    img_name = os.path.splitext(filename)[0] # -.txt
    submission_line.append(img_name)

    if os.path.getsize(os.path.join(exp_dir, filename)) != 0: # if the txt file is not empty (but if there are no annotations you would expect no txt files at all right?)
        # ordered pairs of class (cumulative_prob) prganised by highest prob
        seen_classes = {}
        # cumulative_prob = 0.0
        with open(os.path.join(exp_dir, filename), "r") as label_txt:
            for line in label_txt:
                    # print(line)
                    yolo_pred = line.split() 
                    class_value = int(yolo_pred[0])+1 # we increment with 1 as yolo starts from 0 but fathomnet from 1
                    coords = yolo_pred[1:5]
                    confidence = float(yolo_pred[5])
                    # print(class_value)
                    # print(coords)
                    # print(confidence)
                    
                    # check if this class was seen before
                    if class_value not in seen_classes.keys():
                        seen_classes[class_value] = confidence 
                    else:
                        seen_classes[class_value] = seen_classes[class_value] + confidence # sum all the confidence values per cat

        # print(seen_classes)
        # print(seen_classes.items()) # tuples
        # print(sorted(seen_classes.items(), key=lambda x:x[1])) # sorting on second element of each tuple probability ascending (from low to high)
        sorted_classes_by_prob = sorted(seen_classes.items(), key=lambda x:x[1])
        ordered_classes = ""
        for pair in sorted_classes_by_prob:
            # print(pair)
            ordered_classes = ordered_classes + str(pair[0]) + " "
        ordered_classes = ordered_classes[:-1] # remove final spacebar
        # print(ordered_classes)
        submission_line.append(ordered_classes)
        # print(submission_line)

        #obtain the bbox size probability per image
        labels_txt = open(os.path.join(exp_dir, filename), "r") # the complete txt file
        probs_of_bbs = fathomnet_calculate_prob_of_bb.probability_of_sample(labels_txt)
        # probs_of_bbs is a list containing probabilities for each predicted bbox
        # if its sizes belonging to the sizes of the distribution of the training bbox sizes
        # print('probs_of_bbs', probs_of_bbs) 
        # average probs to get instead for each box only one prob value per image 
        average_bbs_prob=sum(probs_of_bbs)/len(probs_of_bbs)

        #obtain the brightness probability per image
        img_path = os.path.join(img_test_dir, img_name + '.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        avg_brightness = cv2.mean(img)[0]
        # print('avg_brightness', avg_brightness)
        prob_at_brightness = brightness_value_prob.probability_of_sample(avg_brightness)
        # print('prob_at_brightness', prob_at_brightness)

        #calculate bbs and brightness combined
        prob_bbs_and_brightness = (average_bbs_prob + prob_at_brightness)/2

        submission_line.append(prob_bbs_and_brightness) #prob_bbs_and_brightness, prob_at_brightness, average_bbs_prob

    else:
        submission_line.append(0)
        img_path = os.path.join(img_test_dir, img_name + '.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        avg_brightness = cv2.mean(img)[0]
        prob_at_brightness = brightness_value_prob.probability_of_sample(avg_brightness)
        submission_line.append(prob_at_brightness)

    submission_lines.append(submission_line)

with open(csv_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'categories', 'osd'])
    writer.writerows(submission_lines)
