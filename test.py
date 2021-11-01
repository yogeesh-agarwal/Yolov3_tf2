from Yolov3_tf2.postprocessing import np_postprocessing as post_processing
from Yolov3_tf2 import utils as comman_utils
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
batch_size = 1
num_batch = 1
input_size = 416
base_grid_size = 13
grid_scales = [1,2,4]
labels = ["Face" , "Non_Face"]
darknet53_bn = "./data/bn_weights.pickle"
anchors = (anchors * input_size).reshape(9 , 2)
darknet53_weights = "./data/conv_weights.pickle"
sorted_anchors = comman_utils.sort_anchors(anchors)
data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images/"

def create_model(chkpnt_dir):
    latest_chkpnt = tf.train.latest_checkpoint(chkpnt_dir)
    model = yolov3.Yolov3(base_grid_size , len(labels) ,grid_scales , darknet53_weights ,darknet53_bn ,sorted_anchors)
    model.load_weights(latest_chkpnt)
    return model

def predict(aug_image , image , model):
    predictions = model(aug_image , training = False)
    box_objects = post_processing.post_process(predictions , sorted_anchors , images = [image] , show_image = True , class_file = "./data/classnames.txt")
    print(box_objects)

if __name__ == "__main__":
    image = cv2.imread("./data/0_Parade_marchingband_1_849.jpg")
    # image = cv2.imread("./data/0_Parade_Parade_0_904.jpg")
    image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image , (416,416))
    aug_image = np.array(image).reshape(1,416,416,3) / 255.
    yolov3 = create_model("./saved_models")
    predict(aug_image , image , yolov3)
