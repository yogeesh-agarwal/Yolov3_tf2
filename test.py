from Yolov3_tf2.postprocessing import np_postprocessing as post_processing
from Yolov3_tf2.preprocessing.preprocessing import DataGenerator
from Yolov3_tf2 import utils as comman_utils
from Yolov3_tf2.loss.loss import Yolov3Loss
from Yolov3_tf2.model import yolov3
import tensorflow as tf
import numpy as np
import cv2

anchors = np.array([[0.17028831 , 0.35888521],
                    [0.05563053 , 0.09101727],
                    [0.11255733 , 0.21961425],
                    [0.0347448  , 0.06395953],
                    [0.32428802 , 0.42267646],
                    [0.47664651 , 0.65827237],
                    [0.21481797 , 0.20969635],
                    [0.07297461 , 0.14739788],
                    [0.11702667 , 0.11145465]] , dtype = np.float32)

num_batch = 1
is_norm = True
batch_size = 1
input_size = 416
is_augment = False
base_grid_size = 13
grid_scales = [1,2,4]
labels = ["Face" , "Non_Face"]
darknet53_bn = "./data/bn_weights.pickle"
anchors = (anchors * input_size).reshape(9 , 2)
darknet53_weights = "./data/conv_weights.pickle"
gt_data_file = "./data/wider_training_data.pickle"
gt_images_file = "./data/wider_images_names.pickle"
sorted_anchors = comman_utils.sort_anchors(anchors)
data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images/"

def create_datagen():
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
    return batch_generator

def create_model(chkpnt_dir):
    latest_chkpnt = tf.train.latest_checkpoint(chkpnt_dir)
    model = yolov3.Yolov3(sorted_anchors ,
                           len(labels) ,
                           grid_scales ,
                           base_grid_size ,
                           load_pretrain = False ,
                           weights_path =  darknet53_weights ,
                           bn_weights_path = darknet53_bn)
    model.load_weights(latest_chkpnt)
    return model

def create_loss_objects():
    large_loss = Yolov3Loss(base_grid_size , grid_scales[0] , 2 , sorted_anchors[:3])
    medium_loss = Yolov3Loss(base_grid_size , grid_scales[1] , 2 , sorted_anchors[3:6])
    small_loss = Yolov3Loss(base_grid_size , grid_scales[2] , 2 , sorted_anchors[6:])
    return [large_loss, medium_loss, small_loss]

def predict_loss(model):
    batch_generator = create_datagen()
    loss_objects = create_loss_objects()
    for index in range(num_batch):
        batch_images , batch_labels = batch_generator.__getitem__(index)
        predictions = model(batch_images , training = False)
        temp_pred = {"large_scale_preds" : batch_labels[0] , "medium_scale_preds" : batch_labels[1] , "small_scale_preds" : batch_labels[2]}
        for key in temp_pred:
            print(np.amax(abs(predictions[key].numpy() - temp_pred[key])))

        box_objects = post_processing.post_process(predictions , sorted_anchors , show_image = False , class_file = "./data/classnames.txt")

        curr_large_loss = loss_objects[0].call(batch_labels[0] , predictions["large_scale_preds"])
        curr_medium_loss = loss_objects[1].call(batch_labels[1] , predictions["medium_scale_preds"])
        curr_small_loss = loss_objects[2].call(batch_labels[2] , predictions["small_scale_preds"])
        total_loss = curr_large_loss + curr_medium_loss + curr_small_loss
        print(curr_large_loss , curr_medium_loss , curr_small_loss , total_loss)
        # print(box_objects)

def predict(aug_image , image , model):
    predictions = model(aug_image, training = False)
    box_objects = post_processing.post_process(predictions , sorted_anchors , images = [image] , show_image = True , class_file = "./data/classnames.txt")
    # print(box_objects)

if __name__ == "__main__":
    # image = cv2.imread("./data/0_Parade_marchingband_1_849.jpg")
    image = cv2.imread("./data/0_Parade_Parade_0_904.jpg")
    image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image , (416,416))
    aug_image = np.array(image).reshape(1,416,416,3) / 255.
    yolov3 = create_model("./saved_models")
    predict(aug_image , image , yolov3)
    # predict_loss(yolov3)
