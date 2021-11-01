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

def modify_locs(raw_locs , scale_anchors , gt = False):
    modified_localizations = []
    for i in range(len(raw_locs)):
        localizations = raw_locs[i]
        this_scale_anchor = tf.cast(tf.reshape(scale_anchors[i] , [1,1,1,3,2]) , tf.float32)
        modified_loc = pp_utils.modify_locs_util(localizations , this_scale_anchor , ground_truth = gt)
        modified_localizations.append(modified_loc)
    all_locs = tf.concat(modified_localizations , axis = 1)
    return all_locs

def nms(locs , probs , threshold = 0.5):
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
            keep_indcs = nms(tf.squeeze(tf.gather(locs , index_per_class , axis = 0)) , tf.squeeze(tf.gather(probs , index_per_class)))
            for i in tf.range(tf.shape(keep_indcs)[0]):
                if tf.cast(keep_indcs[i] , tf.bool):
                    final_boxes = final_boxes.write(final_boxes.size() , locs[index_per_class[i][0]])
                    final_probs = final_probs.write(final_probs.size() , probs[index_per_class[i][0]])
                    final_class = final_class.write(final_class.size() , c)

    return [final_boxes.stack() , final_probs.stack() , final_class.stack()]

def draw_predictions(img , boxes , probs , classes , class_names , show_image):
    for i in range(len(boxes)):
        curr_box = boxes[i]
        curr_prob = probs[i]
        xmin = int(curr_box[0] - curr_box[2]*0.5)
        ymin = int(curr_box[1] - curr_box[3]*0.5)
        xmax = int(curr_box[0] + curr_box[2]*0.5)
        ymax = int(curr_box[1] + curr_box[3]*0.5)
        conf = str(tf.round(curr_prob , 2) * 100)
        pred_class = classes[i]
        label = class_names[pred_class]
        cv2.rectangle(img , (xmin , ymin) , (xmax , ymax) , (0 , 0 , 255) , 2)
        cv2.putText(img, label + " " + conf + "%" , (xmin , ymin) , cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 0, 0), 2)

    if show_image:
        cv2.imshow("inference_img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def gen_box_objects(final_boxes , final_probs , final_class , num_classes = 2 , gt = False):
    boxes_this_image = []
    for i in range(len(final_boxes)):
        curr_box = final_boxes[i]
        curr_conf = final_probs[i]
        curr_class = final_class[i]
        box = pp_utils.TFBoundingBox(curr_box[0] , curr_box[1],
                                     curr_box[2] , curr_box[3],
                                     curr_conf ,
                                     curr_class ,
                                     False ,
                                     center = False ,
                                     gt = gt)

        boxes_this_image.append(box)
    return boxes_this_image

    # boxes_this_image = []
    # for i in range(len(final_boxes)):
    #     box = comman_utils.BoundingBox(final_boxes[i][0],
    #                       final_boxes[i][1],
    #                       final_boxes[i][2],
    #                       final_boxes[i][3],
    #                       center = True,
    #                       gt = gt)
    #     label = final_class[i]
    #     conf = final_probs[i]
    #     box.add_class(label)
    #     box.add_confidence(conf)
    #
    #     boxes_this_image.append(box)
    # return np.array(boxes_this_image)

    # filename = "../data/box_files/img_{}".format(index)
    # for i in range(len(final_boxes)):
    #     summed_tensor = [final_boxes[i] , final_probs[i] , final_class[i]]
    #     summed_string = tf.strings.format("{}\n{}\n{}\n************\n" , summed_tensor)
    #     tf.io.write_file(filename , summed_string)

    # boxes_this_image = []
    # for i in range(final_boxes.shape[0]):
    #     curr_box = [final_boxes[i][0] , final_boxes[i][1] , final_boxes[i][2] , final_boxes[i][3]]
    #     curr_box.append(final_probs[i])
    #     curr_box.append(final_class[i])
    #     boxes_this_image.append(np.array(curr_box))
    # return boxes_this_image


@tf.function()
# @tf.autograph.experimental.do_not_convert
def post_process(predictions,
                 anchors,
                 images = None,
                 num_classes = 2,
                 class_file = None,
                 show_image = False,
                 image_names = None,
                 ground_truth = None,
                 show_output = False):

    filter_threshold = 0.45
    num_images = tf.shape(predictions[0])[0]
    scale_anchors = [anchors[0:3] , anchors[3:6] , anchors[6:9]]
    for i in range(3):
      predictions[i] = tf.cast(predictions[i] , tf.float32)

    # all_predictions split shape = [[batch_size , 13*13*3 , 7] , [batch_size , 26*26*3 , 7] , [batch_size , 52*52*3 , 7]]
    # all_predictions shape = [batch_size , 10647 , 7]
    all_predictions = modify_locs(predictions , scale_anchors)
    bbox_delta , confidence_scores , cls_pred = tf.split(all_predictions , [4,1,2] , axis = -1)
    adjusted_bbox_delta = transform_bbox(bbox_delta)
    combined_probs = cls_pred * confidence_scores
    det_probs = tf.reduce_max(combined_probs , axis = -1)
    det_class = tf.argmax(combined_probs , axis = -1)

    # if gt are not none, create eval bounding boxes for metric cal.
    if ground_truth is not None:
        all_gt = modify_locs(ground_truth , scale_anchors , gt = True)
        gt_bbox , gt_conf , gt_cls = tf.split(all_gt , [4,1,2] , axis = -1)
        adjusted_gt_box = transform_bbox(gt_bbox)
        combined_gt_probs = gt_cls * gt_conf
        gt_det_probs = tf.reduce_max(combined_gt_probs , axis = -1)
        gt_det_class = tf.argmax(combined_gt_probs , axis = -1)

    pred_box_objects = []
    gt_box_objects = []
    tf.print("num_images" , num_images)
    for i in tf.range(num_images):
        final_boxes , final_probs , final_class = filter_predictions(adjusted_bbox_delta[i] , det_probs[i] , det_class[i] , num_classes)
        keep_index = tf.TensorArray(tf.int32 , size = 0 , dynamic_size = True)
        for j in tf.range(tf.shape(final_probs)[0]):
          if final_probs[j] > filter_threshold:
            keep_index = keep_index.write(keep_index.size() , j)

        final_boxes = tf.gather(final_boxes , keep_index.stack())
        final_probs = tf.gather(final_probs , keep_index.stack())
        final_class = tf.gather(final_class , keep_index.stack())
        # generate the eval bounding boxes (used to calculate metrics.)
        pred_box_objects.append(gen_box_objects(final_boxes, final_probs , final_class))
        # gen_box_objects(final_boxes, final_probs , final_class , i)


        # show the images with bounding boxes.
        if show_image:
            if class_file is None:
                raise Exception("Please provide a file path containing the class names by index")
            if images is None or images[i] is None:
                raise Exception("Please provide the input images to show the detections")
            class_names = pp_utils.load_class_names(class_file)
            draw_predictions(images[i] , final_boxes , final_probs , final_class , class_names , show_image)

        if ground_truth is not None:
            gt_final_boxes , gt_final_probs , gt_final_class = filter_predictions(adjusted_gt_box[i] , gt_det_probs[i] , gt_det_class[i] , num_classes)
            gt_keep_index = tf.TensorArray(tf.int32 , size = 0 , dynamic_size = True)
            for j in tf.range(tf.shape(gt_final_probs)[0]):
              if gt_final_probs[j] > filter_threshold:
                gt_keep_index = gt_keep_index.write(gt_keep_index.size() , j)

            gt_final_boxes = tf.gather(gt_final_boxes , gt_keep_index.stack())
            gt_final_probs = tf.gather(gt_final_probs , gt_keep_index.stack())
            gt_final_class = tf.gather(gt_final_class , gt_keep_index.stack())
            gt_box_objects.append(gen_box_objects(gt_final_boxes, gt_final_probs , gt_final_class))
            # gen_box_objects(gt_final_boxes, gt_final_probs , gt_final_class , i)

    # box_objects = [pred_box_objects]
    # if ground_truth is not None:
    #     box_objects.append(gt_box_objects)
    #
    # return  tf.ragged.constant(box_objects)
    return pred_box_objects
