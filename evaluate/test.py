from Yolov3_tf2.metrics import utils as metric_utils
from Yolov3_tf2.postprocessing import tf_postprocessing as post_processing
from Yolov3_tf2.preprocessing.preprocessing import DataGenerator
from Yolov3_tf2.postprocessing import tf_utils as pp_utils
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
        self.num_batch = 1
        self.is_norm = True
        self.batch_size = 10
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
        self.class_names = pp_utils.load_class_names("../data/classnames.txt")
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

    def predict(self , batch_generator , model ,show_out = False):
        pred_box_list = []
        gt_box_list = []
        all_images = []
        for index in range(self.num_batch):
            batch_images , org_images , batch_labels , org_labels , _ = batch_generator.load_data_for_test(index)
            predictions_dict = model(batch_images, training = False)
            large_scale_preds = predictions_dict["large_scale_preds"]
            medium_scale_preds = predictions_dict["medium_scale_preds"]
            small_scale_preds = predictions_dict["small_scale_preds"]
            predictions = [large_scale_preds , medium_scale_preds , small_scale_preds]
            for img_i in range(batch_images.shape[0]):
                boxes_this_image = post_processing.post_process([prediction[img_i:img_i+1] for prediction in predictions] ,
                                                                self.sorted_anchors)
                # boxes_this_image = post_processing.post_process([batch_label[img_i:img_i+1] for batch_label in batch_labels] ,
                #                                                 self.sorted_anchors)
                pred_box_list.append(boxes_this_image)
                gt_box_list.append(org_labels[img_i])
                all_images.append(org_images[img_i])
                if show_out:
                    pp_utils.draw_predictions(org_images[img_i] , boxes_this_image.numpy()[0] , self.class_names)

        return gt_box_list , pred_box_list , all_images

    def write_boxes(self , pred_box_list , gt_box_list , all_images):
        if len(pred_box_list) != len(gt_box_list):
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

        gt_box_objects = metric_utils.gen_box_objects(gt_box_list , gt = True)
        pred_box_objects = metric_utils.gen_box_objects(pred_box_list)
        for index in range(len(pred_box_list)):
            gt_path = self.gt_folder + "image_{}.txt".format(index)
            det_path = self.detections_folder + "image_{}.txt".format(index)
            comman_utils.convert_to_file(gt_box_objects[index] , pred_box_objects[index] , gt_path , det_path)
            print("image {} boxes have been written".format(index))

    def evaluate(self , mAP = False):
        yolov3 = self.create_model("../saved_models")
        batch_generator = self.create_datagen()
        gt_box_objects , pred_box_objects , all_images = self.predict(batch_generator , yolov3 , show_out = False)
        if mAP:
            self.write_boxes(pred_box_objects , gt_box_objects ,all_images)

if __name__ == "__main__":
    cal_map = True
    evaluator = Evaluate()
    evaluator.evaluate(cal_map)
