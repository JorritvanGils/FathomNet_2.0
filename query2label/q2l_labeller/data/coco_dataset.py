import os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
from tqdm import tqdm

category_map = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "11": 11,
    "13": 12,
    "14": 13,
    "15": 14,
    "16": 15,
    "17": 16,
    "18": 17,
    "19": 18,
    "20": 19,
    "21": 20,
    "22": 21,
    "23": 22,
    "24": 23,
    "25": 24,
    "27": 25,
    "28": 26,
    "31": 27,
    "32": 28,
    "33": 29,
    "34": 30,
    "35": 31,
    "36": 32,
    "37": 33,
    "38": 34,
    "39": 35,
    "40": 36,
    "41": 37,
    "42": 38,
    "43": 39,
    "44": 40,
    "46": 41,
    "47": 42,
    "48": 43,
    "49": 44,
    "50": 45,
    "51": 46,
    "52": 47,
    "53": 48,
    "54": 49,
    "55": 50,
    "56": 51,
    "57": 52,
    "58": 53,
    "59": 54,
    "60": 55,
    "61": 56,
    "62": 57,
    "63": 58,
    "64": 59,
    "65": 60,
    "67": 61,
    "70": 62,
    "72": 63,
    "73": 64,
    "74": 65,
    "75": 66,
    "76": 67,
    "77": 68,
    "78": 69,
    "79": 70,
    "80": 71,
    "81": 72,
    "82": 73,
    "84": 74,
    "85": 75,
    "86": 76,
    "87": 77,
    "88": 78,
    "89": 79,
    "90": 80,
}


class CoCoDataset(data.Dataset):
    """Custom dataset that will load the COCO 2014 dataset and annotations

    This module will load the basic files as provided here: https://cocodataset.org/#download
    If the labels file does not exist yet, it will be created with the included
    helper functions. This class was largely taken from Shilong Liu's repo at
    https://github.com/SlongLiu/query2labels/blob/main/lib/dataset/cocodataset.py.

    Attributes:
        coco (torchvision dataset): Dataset containing COCO data.
        category_map (dict): Mapping of category names to indices.
        input_transform (list of transform objects): List of transforms to apply.
        labels_path (str): Location of labels (if they exist).
        used_category (int): Legacy var.
        labels (list): List of labels.

    """

    def __init__(
        self,
        image_dir,
        anno_path,
        input_transform=None,
        labels_path=None,
        used_category=-1,
    ):
        """Initializes dataset

        Args:
            image_dir (str): Location of COCO images.
            anno_path (str): Location of COCO annotation files.
            input_transform (list of Transform objects, optional): List of transforms to apply.  Defaults to None.
            labels_path (str, optional): Location of labels. Defaults to None.
            used_category (int, optional): Legacy var. Defaults to -1.
        """
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        # with open('./data/coco/category.json','r') as load_category:
        #     self.category_map = json.load(load_category)
        self.category_map = category_map
        self.input_transform = input_transform
        self.labels_path = labels_path
        self.used_category = used_category

        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            l = len(self.coco)
            for i in tqdm(range(l)):
                item = self.coco[i]
                # print(i)
                categories = self.getCategoryList(item[1])
                label = self.getLabelVector(categories)
                self.labels.append(label)
            self.save_datalabels(labels_path)
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        input = self.coco[index][0]
        if self.input_transform:
            input = self.input_transform(input)
        return input, self.labels[index]

    def getCategoryList(self, item):
        """Turns iterable item into list of categories

        Args:
            item (iterable): Any iterable type that contains categories

        Returns:
            list: Categories
        """
        categories = set()
        for t in item:
            categories.add(t["category_id"])
        return list(categories)

    def getLabelVector(self, categories):
        """Creates multi-hot vector for item labels

        Args:
            categories (list): List of categories matching an item

        Returns:
            ndarray: Multi-hot vector for item labels
        """
        label = np.zeros(80)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0  # / label_num
        return label

    def __len__(self):
        return len(self.coco)

    def save_datalabels(self, outpath):
        """Saves datalabels to disk for faster loading next time.

        Args:
            outpath (str): Location where labels are to be saved.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)
