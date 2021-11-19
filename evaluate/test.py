from Yolov3_tf2.postprocessing import np_postprocessing as post_processing
from Yolov3_tf2.preprocessing.preprocessing import DataGenerator
from Yolov3_tf2 import utils as comman_utils
from Yolov3_tf2.loss.loss import Yolov3Loss
from Yolov3_tf2.model import yolov3
import tensorflow as tf
import numpy as np
import shutil
import cv2
import os

class Evaluate(object):
    def __init__(self):
        self.num_batch = 2
        self.is_norm = True
        self.batch_size = 5
        self.input_size = 416
        self.is_augment = False
        self.base_grid_size = 13
        self.grid_scales = [1,2,4]
        self.load_pretrain = False
        self.labels = ["Face" , "Non_Face"]
        self.gt_folder = "../data/ground_truths/"
        self.detections_folder = "../data/detections/"
        self.darknet53_bn = "../data/bn_weights.pickle"
        self.darknet53_weights = "../data/conv_weights.pickle"
        self.gt_data_file = "../data/wider_validation_data.pickle"
        self.gt_images_file = "../data/wider_val_images_names.pickle"
        self.data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_val/WIDER_val/images/"

        self.anchors = np.array([[0.17028831 , 0.35888521],
                            [0.05563053 , 0.09101727],
                            [0.11255733 , 0.21961425],
                            [0.0347448  , 0.06395953],
                            [0.32428802 , 0.42267646],
                            [0.47664651 , 0.65827237],
                            [0.21481797 , 0.20969635],
                            [0.07297461 , 0.14739788],
                            [0.11702667 , 0.11145465]] , dtype = np.float32) * self.input_size
        self.anchors = self.anchors.reshape(9,2)
        self.sorted_anchors = comman_utils.sort_anchors(self.anchors)

    def create_datagen(self):
        batch_generator = DataGenerator(self.input_size ,
                                        self.base_grid_size ,
                                        self.grid_scales ,
                                        self.sorted_anchors ,
                                        self.data_path ,
                                        self.gt_data_file ,
                                        self.gt_images_file ,
                                        self.labels ,
                                        self.is_norm ,
                                        self.is_augment ,
                                        self.batch_size)
        return batch_generator

    def create_model(self , chkpnt_dir):
        latest_chkpnt = tf.train.latest_checkpoint(chkpnt_dir)
        model = yolov3.Yolov3(self.sorted_anchors ,
                               len(self.labels) ,
                               self.grid_scales ,
                               self.base_grid_size ,
                               load_pretrain = self.load_pretrain ,
                               weights_path =  self.darknet53_weights ,
                               bn_weights_path = self.darknet53_bn)
        model.load_weights(latest_chkpnt)
        return model

    def predict(self , batch_generator , model , process_gt = False  ,show_out = False):
        gt_box_objects = []
        pred_box_objects = []
        for index in range(self.num_batch):
            batch_images , org_images , batch_labels , _ = batch_generator.load_data_for_test(index)
            predictions = model(batch_images, training = False)
            print("model prediction completed for batch {}".format(index))
            box_objects = post_processing.post_process(predictions ,
                                                       self.sorted_anchors ,
                                                       ground_truth = None if not process_gt else batch_labels ,
                                                       images = org_images ,
                                                       show_image = show_out ,
                                                       class_file = "../data/classnames.txt")
            for pred_bo in box_objects[0]:
                pred_box_objects.append(pred_bo)
            if process_gt:
                for gt_bo in box_objects[1]:
                    gt_box_objects.append(gt_bo)

            print(len(gt_box_objects))

        return [pred_box_objects] if not process_gt else [pred_box_objects , gt_box_objects]

    def write_boxes(self , pred_box_objects , gt_box_objects):
        if len(pred_box_objects) != len(gt_box_objects):
            raise Exception("pred_box_objects length should match with gt_box_objects length")

        if os.path.isdir(self.detections_folder):
            shutil.rmtree(self.detections_folder)
            os.makedirs(self.detections_folder)
        else:
            os.makedirs(self.detections_folder)
        if os.path.isdir(self.gt_folder):
            shutil.rmtree(self.gt_folder)
            os.makedirs(self.gt_folder)
        else:
            os.makedirs(self.gt_folder)
        for index in range(len(pred_box_objects)):
            gt_path = self.gt_folder + "image_{}.txt".format(index)
            det_path = self.detections_folder + "image_{}.txt".format(index)
            comman_utils.convert_to_file(gt_box_objects[index] , pred_box_objects[index] , gt_path , det_path)
            print("image {} boxes have been written".format(index))

    def evaluate(self , mAP = False):
        yolov3 = self.create_model("../saved_models")
        batch_generator = self.create_datagen()
        box_objects = self.predict(batch_generator , yolov3 , show_out = False , process_gt = True)
        if len(box_objects) == 2:
            self.write_boxes(box_objects[0] , box_objects[1])



if __name__ == "__main__":
    # image = cv2.imread("../data/0_Parade_marchingband_1_849.jpg")
    # # image = cv2.imread("../data/0_Parade_Parade_0_904.jpg")
    # # image = cv2.imread("../data/2_Demonstration_Demonstration_Or_Protest_2_2.jpg")
    # image = cv2.imread("/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_test/images/0--Parade/0_Parade_marchingband_1_250.jpg")
    # image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image , (416,416))
    # aug_image = np.array(image).reshape(1,416,416,3) / 255.
    # yolov3 = create_model("../saved_models")
    # predict(aug_image , image , yolov3)

    evaluator = Evaluate()
    evaluator.evaluate()
