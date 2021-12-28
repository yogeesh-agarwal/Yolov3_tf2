import os
import cv2
import json
import pickle
import numpy as np
import tensorflow as tf
from Yolov3_tf2 import utils as  comman_utils
from Yolov3_tf2.postprocessing import tf_utils as pp_utils

# function to ensure the predictions are under the image boundaries.
def transform_bbox(bbox , img_shape = [416 , 416]):
    box_split = tf.split(bbox , [1,1,1,1] , axis = -1)
    x_center  , y_center , w , h = box_split[0] , box_split[1] , box_split[2] , box_split[3]
    xmin = x_center - w/2
    ymin = y_center - h/2
    xmax = x_center + w/2
    ymax = y_center + h/2

    xmin = tf.minimum(tf.maximum(0.0 , xmin) , img_shape[0]-1.0)
    ymin = tf.minimum(tf.maximum(0.0 , ymin) , img_shape[1]-1.0)
    xmax = tf.maximum(tf.minimum(img_shape[0]-1.0 , xmax) , 0.0)
    ymax = tf.maximum(tf.minimum(img_shape[1]-1.0 , ymax) , 0.0)

    adjusted_w = xmax - xmin + 1.0
    adjusted_h = ymax - ymin + 1.0
    adjusted_x_center = xmin + (0.5 * adjusted_w)
    adjusted_y_center = ymin + (0.5 * adjusted_h)
    return tf.concat([adjusted_x_center , adjusted_y_center , adjusted_w , adjusted_h] , axis = -1)

def modify_locs(raw_locs , scale_anchors , img_size = 416 , gt = False):
    grid_scale = [1,2,4]
    modified_localizations = []
    for i in range(len(raw_locs)):
        localizations = raw_locs[i]
        ratio = img_size / (13 * grid_scale[i])
        this_scale_anchor = tf.cast(tf.cast(tf.reshape(scale_anchors[i] , [1,1,1,3,2]) , tf.float32) / ratio , tf.float32)
        modified_loc = pp_utils.modify_locs_util(localizations , this_scale_anchor , ground_truth = gt)
        modified_localizations.append(modified_loc)
    all_locs = tf.concat(modified_localizations , axis = 1)
    return all_locs

def nms(locs , probs , threshold = 0.3):
    # convert the scalar probs to 1 rank tensor (0 dimension tensor in argsort function bug.)
    rank = tf.rank(probs)
    probs = tf.cond(rank==0 , lambda : tf.expand_dims(probs , axis = 0) , lambda : probs)
    pred_order = tf.argsort(probs)[::-1]
    new_locs = tf.gather(locs , pred_order , axis = 0)
    new_porbs = tf.gather(probs , pred_order)
    keep = tf.TensorArray(tf.bool , size = len(pred_order))
    for i in tf.range(len(pred_order)):
        keep = keep.write(i , True)

    for i in range(len(pred_order)-1):
        overlaps = pp_utils.iou(new_locs[i+1 : ] , new_locs[i])
        for j in range(len(overlaps)):
            if overlaps[j] > threshold:
                keep = keep.write(i+j+1 , False)
    return keep.stack()

def filter_predictions(localizations , det_probs_this_inst , det_class_this_inst , num_classes):
    # det_probs_this_inst.shape : 10647 , localizations.shape : (10647 , 4)
    final_boxes = tf.TensorArray(tf.float32 , size = 0 , dynamic_size = True , name = "final_boxes")
    final_probs = tf.TensorArray(tf.float32 , size = 0 , dynamic_size = True , name = "final_probs")
    final_class = tf.TensorArray(tf.int32 , size = 0 , dynamic_size = True , name = "final_class")
    max_prediction = 100
    if max_prediction < len(det_probs_this_inst):
        pred_order = tf.argsort(det_probs_this_inst)[ : -max_prediction-1:-1]
        locs = tf.cast(tf.gather(localizations , pred_order , axis = 0) , tf.float32)
        probs = tf.cast(tf.gather(det_probs_this_inst , pred_order) , tf.float32)
        cls_idx = tf.cast(tf.gather(det_class_this_inst , pred_order) , tf.int32)

        for c in tf.range(num_classes):
            index_per_class = tf.cast(tf.where(cls_idx == c) , tf.int32)

            if len(index_per_class) > 0:
                keep_indcs = nms(tf.squeeze(tf.gather(locs , index_per_class , axis = 0)) , tf.squeeze(tf.gather(probs , index_per_class)))
                for i in tf.range(tf.shape(keep_indcs)[0]):
                    if tf.cast(keep_indcs[i] , tf.bool):
                        final_boxes = final_boxes.write(final_boxes.size() , locs[index_per_class[i][0]])
                        final_probs = final_probs.write(final_probs.size() , probs[index_per_class[i][0]])
                        final_class = final_class.write(final_class.size() , c)

    return [final_boxes.stack() , final_probs.stack() , final_class.stack()]

def gen_boxes(final_boxes , final_probs , final_class , num_classes = 2 , gt = False):
    num_boxes = tf.shape(final_boxes)[0]
    final_boxes = tf.cast(tf.reshape(final_boxes , (1 , num_boxes , 4)) , tf.float32)
    final_probs = tf.cast(tf.reshape(final_probs , (1 , num_boxes , 1)) , tf.float32)
    final_class = tf.cast(tf.reshape(final_class , (1 , num_boxes , 1)) , tf.float32)
    return tf.concat([final_boxes , final_probs , final_class] , axis = -1)

@tf.function()
# @tf.autograph.experimental.do_not_convert
def post_process(predictions,
                 anchors,
                 num_classes = 2):

    filter_threshold = 0.7
    scale_anchors = [anchors[0:3] , anchors[3:6] , anchors[6:9]]

    # all_predictions split shape = [[batch_size , 13*13*3 , 7] , [batch_size , 26*26*3 , 7] , [batch_size , 52*52*3 , 7]]
    # all_predictions shape = [batch_size , 10647 , 7]
    all_predictions = modify_locs(predictions , scale_anchors)
    bbox_delta , confidence_scores , cls_pred = tf.split(all_predictions , [4,1,2] , axis = -1)
    adjusted_bbox_delta = transform_bbox(bbox_delta)
    combined_probs = cls_pred * confidence_scores
    det_probs = tf.reduce_max(combined_probs , axis = -1)
    det_class = tf.argmax(combined_probs , axis = -1)

    final_boxes , final_probs , final_class = filter_predictions(adjusted_bbox_delta[0] , det_probs[0] , det_class[0] , num_classes)
    keep_index = tf.TensorArray(tf.int32 , size = 0 , dynamic_size = True)
    for j in tf.range(tf.shape(final_probs)[0]):
      if final_probs[j] > filter_threshold:
        keep_index = keep_index.write(keep_index.size() , j)

    final_boxes = tf.gather(final_boxes , keep_index.stack())
    final_probs = tf.gather(final_probs , keep_index.stack())
    final_class = tf.gather(final_class , keep_index.stack())
    box_objects = gen_boxes(final_boxes , final_probs , final_class)
    return box_objects
