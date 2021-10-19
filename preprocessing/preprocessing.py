import os
import pdb
import cv2
import math
import pickle
import random
import numpy as np
import imgaug as ia
import tensorflow as tf
from Yolov3_tf2 import utils
from operator import itemgetter
import imgaug.augmenters as iaa

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self , input_size ,
                        base_grid_size ,
                        grid_scales ,
                        anchors ,
                        data_path ,
                        instances_file ,
                        image_names ,
                        labels ,
                        is_norm ,
                        is_augment ,
                        batch_size):
        self.shuffle = True
        self.labels = labels
        self.anchors = anchors
        self.is_norm = is_norm
        self.data_path = data_path
        self.batch_size = batch_size
        self.is_augment = is_augment
        self.input_width = input_size
        self.grid_scales = grid_scales
        self.input_height = input_size
        self.num_anchors_per_stage = 3
        self.base_grid_width = base_grid_size
        self.base_grid_height = base_grid_size
        self.image_names = self.load_pickle(image_names)
        self.train_data = self.load_pickle(instances_file)
        self.num_train_instances = len(self.train_data)

        sometimes = lambda aug : iaa.Sometimes(0.55, aug)
        self.augmentor = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            sometimes(iaa.Affine(
                rotate = (-20 , 20),
                shear = (-13 , 13),
            )),
            iaa.SomeOf((0 , 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0 , 3.0)),
                    iaa.AverageBlur(k =  (2 , 7)),
                    iaa.MedianBlur(k = (3 , 11)),
                ]),
                iaa.Sharpen(alpha = (0 , 1.0) , lightness = (0.75 , 1.5)),
                iaa.AdditiveGaussianNoise(loc = 0 , scale = (0.0 , 0.05*255) , per_channel = 0.5),
                iaa.Add((-10 , 10) , per_channel = 0.5),
                iaa.Multiply((0.5 , 1.5) , per_channel = 0.5),
                iaa.LinearContrast((0.5 , 2.0) , per_channel = 0.5),
            ] , random_order = True)
        ] , random_order = True)
        self.on_epoch_end()

    def load_pickle(self , filepath):
        with open(filepath , "rb") as content:
            return pickle.load(content)

    def augment_data(self, images , labels):
        bb_objects = []
        for objects in labels:
            curr_bb_objects = []
            for object in objects:
                curr_bb_objects.append(ia.BoundingBox(object[0] , object[1] , object[0]  + object[2] , object[1] + object[3]))
            bb_objects.append(curr_bb_objects)

        aug_images , aug_bbs = self.augmentor(images = images , bounding_boxes = bb_objects)
        augmented_boxes = []
        augmented_images = []

        for image , bb in zip(aug_images , aug_bbs):
            boxes = []
            for bbox in bb:
                boxes.append([bbox.x1 , bbox.y1 , (bbox.x2 - bbox.x1), (bbox.y2 - bbox.y1)])
            augmented_boxes.append(boxes)
            augmented_images.append(image)
        return augmented_images , augmented_boxes

    def encode_data(self , starting_index , ending_index):
        images = []
        labels = []
        for instances_index in range(starting_index , ending_index):
            image_path = os.path.join(self.data_path , self.image_names[instances_index])
            image = cv2.cvtColor(cv2.imread(image_path) , cv2.COLOR_BGR2RGB)
            image = cv2.resize(image , (self.input_width , self.input_height))

            if self.image_names[instances_index] not in self.train_data:
                raise Exception("image_name not found")
            label = self.train_data[self.image_names[instances_index]]

            resized_label = []
            for index in range(len(label)):
                X = int(label[index][0] * self.input_width)
                Y = int(label[index][1] * self.input_height)
                W = int(label[index][2] * self.input_width)
                H = int(label[index][3] * self.input_height)
                resized_label.append([X , Y , W , H])
            images.append(image)
            labels.append(resized_label)

        augmented_images , augmented_boxes = images , labels
        if self.is_augment:
            augmented_images , augmented_boxes = self.augment_data(images , labels)

        encoded_images = []
        detector_indexes = []
        encoded_labels_large_objects = np.zeros([ending_index - starting_index , self.base_grid_height * self.grid_scales[0] , self.base_grid_width * self.grid_scales[0] , self.num_anchors_per_stage , (4+1+len(self.labels))])
        encoded_labels_medium_objects = np.zeros([ending_index - starting_index , self.base_grid_height * self.grid_scales[1] , self.base_grid_width * self.grid_scales[1] , self.num_anchors_per_stage , (4+1+len(self.labels))])
        encoded_labels_small_objects = np.zeros([ending_index - starting_index , self.base_grid_height * self.grid_scales[2] , self.base_grid_width * self.grid_scales[2] , self.num_anchors_per_stage , (4+1+len(self.labels))])
        for index in range(len(augmented_images)):
            aug_image = augmented_images[index]
            if self.is_norm:
                aug_image = aug_image / 255.

            window_index = []
            for object in augmented_boxes[index]:
                max_iou = -1
                best_anchor_index = 0
                dummy_box = utils.BoundingBox(0 , 0 ,object[2] , object[3])
                for anchor_index in range(len(self.anchors)):
                    anchor_box = utils.BoundingBox(0 , 0 , self.anchors[anchor_index][0] , self.anchors[anchor_index][1])
                    iou = dummy_box.iou(anchor_box)
                    if iou > max_iou:
                        max_iou = iou
                        best_anchor_index = anchor_index

                window_index.append(best_anchor_index)
                yolo_id = best_anchor_index // self.num_anchors_per_stage
                curr_grid_width = self.base_grid_width * self.grid_scales[yolo_id]
                curr_grid_height = self.base_grid_height * self.grid_scales[yolo_id]
                center_x = object[0] + object[2] * 0.5
                center_y = object[1] + object[3] * 0.5
                center_x = center_x / (float(self.input_width) / curr_grid_width)
                center_y = center_y / (float(self.input_height) / curr_grid_height)
                center_w = np.log(object[2] / float(self.anchors[best_anchor_index][0]))
                center_h = np.log(object[3] / float(self.anchors[best_anchor_index][1]))
                curr_grid_x = min(int(math.floor(center_x)) , curr_grid_width - 1)
                curr_grid_y = min(int(math.floor(center_y)) , curr_grid_height - 1)
                if yolo_id == 0:
                    encoded_labels_large_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 0:4] = [center_x , center_y , center_w , center_h]
                    encoded_labels_large_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 4] = 1
                    encoded_labels_large_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 5] = 1
                    encoded_labels_large_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 6] = 0
                elif yolo_id == 1:
                    encoded_labels_medium_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 0:4] = [center_x , center_y , center_w , center_h]
                    encoded_labels_medium_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 4] = 1
                    encoded_labels_medium_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 5] = 1
                    encoded_labels_medium_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 6] = 0
                elif yolo_id == 2:
                    encoded_labels_small_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 0:4] = [center_x , center_y , center_w , center_h]
                    encoded_labels_small_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 4] = 1
                    encoded_labels_small_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 5] = 1
                    encoded_labels_small_objects[index , curr_grid_y , curr_grid_x , best_anchor_index % self.num_anchors_per_stage , 6] = 0
                else:
                    raise Exception("selected yolo id based on best anchor is not valid , " , yolo_id)

            detector_indexes.append(window_index)
            encoded_images.append(aug_image)

        return encoded_images , encoded_labels_large_objects , encoded_labels_medium_objects , encoded_labels_small_objects , detector_indexes

    def on_epoch_end(self):
        if self.shuffle:
            # pdb.set_trace()
            np.random.shuffle(self.image_names)

    def load_data(self  , index):
        starting_index = index * self.batch_size
        ending_index = starting_index + self.batch_size
        if ending_index > len(self.train_data):
            ending_index = len(self.train_data)
            starting_index = ending_index - self.batch_size

        encoded_images , encoded_labels_large_objects , encoded_labels_medium_objects , encoded_labels_small_objects , detector_indexes = self.encode_data(starting_index , ending_index)
        return encoded_images, [encoded_labels_large_objects, encoded_labels_medium_objects , encoded_labels_small_objects , detector_indexes]

    def __len__(self):
        return self.num_train_instances

    def __getitem__(self, index):
        encoded_images , encoded_labels = self.load_data(index)
        return encoded_images, encoded_labels
