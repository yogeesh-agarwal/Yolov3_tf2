import cv2
import numpy as np
import tensorflow as tf
from Yolov3_tf2 import utils as comman_utils
from Yolov3_tf2.postprocessing import tf_utils as pp_utils
from Yolov3_tf2.preprocessing.preprocessing import DataGenerator
from Yolov3_tf2.postprocessing.np_postprocessing import post_process

def test_postprocess():
    anchors = np.array([[0.17028831 , 0.35888521],
                        [0.05563053 , 0.09101727],
                        [0.11255733 , 0.21961425],
                        [0.0347448  , 0.06395953],
                        [0.32428802 , 0.42267646],
                        [0.47664651 , 0.65827237],
                        [0.21481797 , 0.20969635],
                        [0.07297461 , 0.14739788],
                        [0.11702667 , 0.11145465]] , dtype = np.float32)

    batch_size = 1
    num_batch = 1
    is_norm = False
    input_size = 416
    is_augment = False
    base_grid_size = 13
    grid_scales = [1,2,4]
    labels = ["Face" , "Non_Face"]
    anchors = (anchors * input_size).reshape(9 , 2)
    sorted_anchors = comman_utils.sort_anchors(anchors)
    gt_data_file = "../data/wider_training_data.pickle"
    gt_images_file = "../data/wider_images_names.pickle"
    data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images/"
    class_names = pp_utils.load_class_names("../data/classnames.txt")
    batch_generator = DataGenerator(416 ,
                                    base_grid_size ,
                                    grid_scales ,
                                    sorted_anchors ,
                                    data_path ,
                                    gt_data_file ,
                                    gt_images_file ,
                                    labels ,
                                    is_norm ,
                                    is_augment ,
                                    batch_size)

    for index in range(num_batch):
        batch_images , batch_labels = batch_generator.__getitem__(index)
        box_objects = post_process(batch_labels , sorted_anchors , images = batch_images , show_image = True , class_file = "../data/classnames.txt")
        print(box_objects)


if __name__ == "__main__":
    test_postprocess()
