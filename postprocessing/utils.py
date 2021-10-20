from Yolov3_tf2 import utils as comman_utils
import numpy as np
import os

def sigmoid(x):
    return 1 / (1 + np.exp(x * -1))

def softmax(x , axis = -1 , t = -100):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x ) * t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis , keepdims = True)

def modify_locs_util(localizations , anchors , img_shape = [416, 416] , ground_truth = False):
    grid_shape = localizations.shape[1:3]
    num_anchors = localizations.shape[3]
    num_classes = localizations.shape[4] - 5
    strides = [img_shape[0] // grid_shape[0], img_shape[1] // grid_shape[1]]
    cell_grid = comman_utils.gen_cell_grid(grid_shape[0] , grid_shape[1] , num_anchors)

    if not ground_truth:
        xy = sigmoid(localizations[... , 0:2]) + cell_grid
        conf = sigmoid(localizations[... , 4])
        classes = softmax(localizations[... , 5:])
    else:
        xy = localizations[... , 0:2]
        conf = localizations[... , 4]
        classes = localizations[... , 5:]

    xy = np.reshape(xy * strides , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 2])
    wh = np.reshape(np.exp(localizations[... , 2:4]) * anchors , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 2])
    conf = np.reshape(conf , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 1])
    classes = np.reshape(classes , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 2])
    modified_locs = np.concatenate([xy , wh , conf , classes] , axis = -1)
    return np.reshape(modified_locs , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 5+num_classes])

def iou(boxes , box):
    boxes_x_min = boxes[... , 0] - boxes[... , 2]*0.5
    boxes_y_min = boxes[... , 1] - boxes[... , 3]*0.5
    boxes_x_max = boxes[... , 0] + boxes[... , 2]*0.5
    boxes_y_max = boxes[... , 1] + boxes[... , 3]*0.5

    ref_x_min = box[0] - box[2]*0.5
    ref_y_min = box[1] - box[3]*0.5
    ref_x_max = box[0] + box[2]*0.5
    ref_y_max = box[1] + box[3]*0.5

    intersected_width = np.maximum(np.minimum(boxes_x_max , ref_x_max) - np.maximum(boxes_x_min , ref_x_min) , 0)
    intersected_height = np.maximum(np.minimum(boxes_y_max , ref_y_max) - np.maximum(boxes_y_min , ref_y_min) , 0)
    intersection = intersected_width * intersected_height
    union = (boxes[... , 2] * boxes[... , 3]) + (box[... , 2] * box[... , 3]) - intersection
    return intersection / union
