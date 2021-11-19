import os
import cv2
import json
import pickle
import numpy as np
from Yolov3_tf2 import utils as  comman_utils
from Yolov3_tf2.postprocessing import tf_utils as pp_utils

# function to ensure the predictions are under the image boundaries.
def transform_bbox(bbox , img_shape = [416 , 416]):
    box_split = np.split(bbox , [1,2,3] , axis = -1)
    x_center  , y_center , w , h = box_split[0] , box_split[1] , box_split[2] , box_split[3]
    xmin = x_center - w/2
    ymin = y_center - h/2
    xmax = x_center + w/2
    ymax = y_center + h/2

    xmin = np.minimum(np.maximum(0.0 , xmin) , img_shape[0]-1.0)
    ymin = np.minimum(np.maximum(0.0 , ymin) , img_shape[1]-1.0)
    xmax = np.maximum(np.minimum(img_shape[0]-1.0 , xmax) , 0.0)
    ymax = np.maximum(np.minimum(img_shape[1]-1.0 , ymax) , 0.0)

    adjusted_w = xmax - xmin + 1.0
    adjusted_h = ymax - ymin + 1.0
    adjusted_x_center = xmin + (0.5 * adjusted_w)
    adjusted_y_center = ymin + (0.5 * adjusted_h)
    return np.concatenate([adjusted_x_center , adjusted_y_center , adjusted_w , adjusted_h] , axis = -1)

def modify_locs(raw_locs , scale_anchors , gt = False):
    grid_scale = [1,2,4]
    modified_localizations = []
    for i in range(len(raw_locs)):
        localizations = raw_locs[i]
        ratio = 416 / (13 * grid_scale[i])
        this_scale_anchor = np.reshape(scale_anchors[i] , [1,1,1,3,2]).astype(np.float32) / ratio
        modified_loc = pp_utils.modify_locs_util(localizations , this_scale_anchor , ground_truth = gt)
        modified_localizations.append(modified_loc)
    all_locs = np.concatenate([modified_localizations[0] , modified_localizations[1] , modified_localizations[2]] , axis = 1)
    return all_locs

def nms(locs , probs , threshold = 0.5):
    pred_order = np.argsort(probs)[::-1]
    new_locs = locs[pred_order]
    new_porbs = probs[pred_order]
    keep = [True] * len(pred_order)

    for i in range(len(pred_order)-1):
        overlaps = pp_utils.iou(new_locs[i+1 : ] , new_locs[i])
        for j in range(len(overlaps)):
            if overlaps[j] > threshold:
                keep[pred_order[i+j+1]] = False
    return keep

def filter_predictions(localizations , det_probs , det_class , num_classes):
    max_prediction = 100
    if max_prediction < len(det_probs):
        pred_order = np.argsort(det_probs)[ : -max_prediction-1:-1]
        locs = localizations[pred_order]
        probs = det_probs[pred_order]
        cls_idx = det_class[pred_order]

        final_boxes = []
        final_probs = []
        final_class = []

        for c in range(num_classes):
            index_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
            keep = nms(locs[index_per_class] , probs[index_per_class])
            for i in range(len(keep)):
                if keep[i]:
                    final_boxes.append(locs[index_per_class[i]])
                    final_probs.append(probs[index_per_class[i]])
                    final_class.append(c)

        return [final_boxes , final_probs , final_class]

def draw_predictions(img , boxes , probs , classes , class_names):
    for i in range(len(boxes)):
        xmin = int(boxes[i][0] - boxes[i][2]*0.5)
        ymin = int(boxes[i][1] - boxes[i][3]*0.5)
        xmax = int(boxes[i][0] + boxes[i][2]*0.5)
        ymax = int(boxes[i][1] + boxes[i][3]*0.5)
        conf = str(round(probs[i] , 2) * 100)
        pred_class = classes[i]
        label = class_names[pred_class]
        cv2.rectangle(img , (xmin , ymin) , (xmax , ymax) , (0 , 0 , 255) , 2)
        cv2.putText(img, label + " " + conf + "%" , (xmin , ymin) , cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 0, 0), 2)

    cv2.imshow("inference_img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gen_box_objects(final_boxes , final_probs , final_class , gt = False):
    boxes_this_image = []
    for i in range(len(final_boxes)):
        box = comman_utils.BoundingBox(final_boxes[i][0],
                          final_boxes[i][1],
                          final_boxes[i][2],
                          final_boxes[i][3],
                          center = True,
                          gt = gt)
        label = final_class[i]
        conf = final_probs[i]
        box.add_class(label)
        box.add_confidence(conf)

        boxes_this_image.append(box)
    return boxes_this_image

def post_process(predictions,
                 anchors,
                 images = None,
                 num_classes = 2,
                 class_file = None,
                 image_names = None,
                 ground_truth = None,
                 show_image = False):

    filter_threshold = 0.65
    scale_anchors = [anchors[0:3] , anchors[3:6] , anchors[6:9]]
    numpy_predictions = [predictions[key].numpy().astype(np.float32) for key in predictions]
    num_images = numpy_predictions[0].shape[0]

    # all_predictions split shape = [[batch_size , 13*13*3 , 7] , [batch_size , 26*26*3 , 7] , [batch_size , 52*52*3 , 7]]
    # all_predictions shape = [batch_size , 10647 , 7]
    all_predictions = modify_locs(numpy_predictions , scale_anchors)
    bbox_delta , confidence_scores , cls_pred = np.split(all_predictions , [4,5] , axis = -1)
    adjusted_bbox_delta = transform_bbox(bbox_delta)
    combined_probs = cls_pred * confidence_scores
    det_probs = np.amax(combined_probs , axis = -1)
    det_class = np.argmax(combined_probs , axis = -1)

    # if gt are not none, create eval bounding boxes for metric cal.
    if ground_truth is not None:
        all_gt = modify_locs(ground_truth , scale_anchors , gt = True)
        gt_bbox , gt_conf , gt_cls = np.split(all_gt , [4,5] , axis = -1)
        adjusted_gt_box = transform_bbox(gt_bbox)
        combined_gt_probs = gt_cls * gt_conf
        gt_det_probs = np.amax(combined_gt_probs , axis = -1)
        gt_det_class = np.argmax(combined_gt_probs , axis = -1)

    pred_box_objects = []
    gt_box_objects = []
    for i in range(num_images):
        final_boxes , final_probs , final_class = filter_predictions(adjusted_bbox_delta[i] , det_probs[i] , det_class[i] , num_classes)
        keep_index = [index for index in range(len(final_probs)) if final_probs[index] > filter_threshold]
        final_boxes = [final_boxes[index] for index in keep_index]
        final_probs = [final_probs[index] for index in keep_index]
        final_class = [final_class[index] for index in keep_index]
        # generate the eval bounding boxes (used to calculate metrics.)
        pred_box_objects.append(gen_box_objects(final_boxes, final_probs , final_class))

        # show the images with bounding boxes.
        if show_image:
            if class_file is None:
                raise Exception("Please provide a file path containing the class names by index")
            if images is None:
                raise Exception("Please provide the input images to show the detections")
            class_names = pp_utils.load_class_names(class_file)
            draw_predictions(images[i] , final_boxes , final_probs , final_class , class_names)

        if ground_truth is not None:
            gt_final_boxes , gt_final_probs , gt_final_class = filter_predictions(adjusted_gt_box[i] , gt_det_probs[i] , gt_det_class[i] , num_classes)
            keep_index = [index for index in range(len(gt_final_probs)) if gt_final_probs[index] > filter_threshold]
            gt_final_boxes = [gt_final_boxes[index] for index in keep_index]
            gt_final_probs = [gt_final_probs[index] for index in keep_index]
            gt_final_class = [gt_final_class[index] for index in keep_index]
            gt_box_objects.append(gen_box_objects(gt_final_boxes, gt_final_probs , gt_final_class , gt = True))
    box_objects = [np.array(pred_box_objects , dtype = object)]
    if ground_truth is not None:
        box_objects.append(np.array(gt_box_objects , dtype = object))
    return  box_objects
