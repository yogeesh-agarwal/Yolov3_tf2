import os
import sys
import numpy as np
import tensorflow as tf
from Yolov3_tf2.model.yolov3 import Yolov3
from Yolov3_tf2.loss.loss import Yolov3Loss
from Yolov3_tf2 import utils as comman_utils
from Yolov3_tf2.preprocessing.preprocessing import DataGenerator

def gen_callbacks(tb_ld , cp_path):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = tb_ld,
                                                          histogram_freq = 1,
                                                          update_freq = "batch")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = cp_path,
                                                             verbose = 1,
                                                             mode = "min",
                                                             save_best_only = True,
                                                             monitor = "train_loss",
                                                             save_weights_only = True)

    reduce_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = "train_loss",
                                                                      factor = 0.5,
                                                                      patience = 50,
                                                                      verbose = 1,
                                                                      mode = "min",
                                                                      min_lr = 1e-6)
    return [tensorboard_callback , checkpoint_callback , reduce_on_plateau_callback]

# def get_epoch_number(latest_chkpnt):
    # chkpnt_name = latest_chkpnt.split(".ckpt")[0]
    # epoch_number = chkpnt_name.split("-")[1]
    # return int(epoch_number)

def get_epoch_number():
    return 0

def train(input_size,
          base_grid_size,
          grid_scales,
          sorted_anchors,
          batch_size,
          num_epochs,
          init_lr,
          logs_dir,
          save_dir,
          train_data_path,
          val_data_path,
          train_file,
          val_file,
          train_image_names,
          val_image_names,
          labels,
          is_norm,
          is_augment,
          darknet53_weights,
          darknet53_bn):

    train_summary_writer = tf.summary.create_file_writer(logs_dir)
    tf.summary.experimental.set_step(0)
    callbacks = gen_callbacks(logs_dir+"/loss" , save_dir+"yolov3.ckpt")
    train_data_generator = DataGenerator(input_size ,
                                         base_grid_size ,
                                         grid_scales ,
                                         sorted_anchors ,
                                         train_data_path ,
                                         train_file ,
                                         train_image_names ,
                                         labels ,
                                         is_norm ,
                                         is_augment,
                                         batch_size)


    val_data_generator = DataGenerator(input_size ,
                                         base_grid_size ,
                                         grid_scales ,
                                         sorted_anchors ,
                                         val_data_path ,
                                         val_file ,
                                         val_image_names ,
                                         labels ,
                                         is_norm ,
                                         is_augment,
                                         batch_size)

    face_detector = Yolov3(base_grid_size , len(labels) ,grid_scales , darknet53_weights ,darknet53_bn ,sorted_anchors)
    large_obj_loss = Yolov3Loss(base_grid_size ,grid_scales[0] , len(labels) ,sorted_anchors[:3], summary_writer = train_summary_writer , name = "large_obj_loss")
    medium_obj_loss = Yolov3Loss(base_grid_size ,grid_scales[1] , len(labels) ,sorted_anchors[3:6], summary_writer = train_summary_writer , name = "medium_obj_loss")
    small_obj_loss = Yolov3Loss(base_grid_size ,grid_scales[2] , len(labels) ,sorted_anchors[6:], summary_writer = train_summary_writer , name = "small_obj_loss")
    epoch_number = 0
    if not os.listdir(save_dir):
        print("No trained model found m starting from scratch")
    else:
        latest_chkpnt = tf.train.latest_checkpoint(save_dir)
        face_detector.load_weights(latest_chkpnt)
        epoch_number = get_epoch_number()
        print("Resuming training with partially trained model from epoch : " , epoch_number)

    face_detector.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = init_lr) ,
                          loss = {"large_scale_preds" : large_obj_loss ,
                                  "medium_scale_preds" : medium_obj_loss ,
                                  "small_scale_preds" : small_obj_loss})

    face_detector.fit(train_data_generator ,
                      epochs = num_epochs,
                      use_multiprocessing = True,
                      initial_epoch = epoch_number,
                      callbacks = callbacks)

def main():
    input_size = 416
    base_grid_size = 13
    grid_scales = [1,2,4]
    batch_size = 1
    num_epochs = 1000
    init_lr = 0.0001
    logs_dir = "../logs/"
    save_dir = "../saved_models/"
    train_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images/"
    train_file = "../data/wider_training_data.pickle"
    train_image_names = "../data/wider_images_names.pickle"
    val_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_val/WIDER_val/images/"
    val_file = "../data/wider_validation_data.pickle"
    val_image_names = "../data/wider_val_images_names.pickle"
    labels = ["Face" , "Non_Face"]
    is_norm = True
    is_augment = False
    darknet53_weights = "../data/conv_weights.pickle"
    darknet53_bn = "../data/bn_weights.pickle"

    anchors = np.array([[0.17028831 , 0.35888521],
                        [0.05563053 , 0.09101727],
                        [0.11255733 , 0.21961425],
                        [0.0347448  , 0.06395953],
                        [0.32428802 , 0.42267646],
                        [0.47664651 , 0.65827237],
                        [0.21481797 , 0.20969635],
                        [0.07297461 , 0.14739788],
                        [0.11702667 , 0.11145465]] , dtype = np.float32).reshape(9,2) * input_size

    sorted_anchors = comman_utils.sort_anchors(anchors)

    train(input_size,
          base_grid_size,
          grid_scales,
          sorted_anchors,
          batch_size,
          num_epochs,
          init_lr,
          logs_dir,
          save_dir,
          train_data_path,
          val_data_path,
          train_file,
          val_file,
          train_image_names,
          val_image_names,
          labels,
          is_norm,
          is_augment,
          darknet53_weights,
          darknet53_bn)

if __name__ == '__main__':
    main()
