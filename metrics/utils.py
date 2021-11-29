import numpy as np
import Yolov3_tf2.utils as comman_utils

def gen_box_objects(boxes , gt = False):
    box_objects = []
    for boxes_per_instance in boxes:
        box_objects_this_instance = []
        if not gt:
            boxes_per_instance = boxes_per_instance[0]
        for box in boxes_per_instance:
            b_object = comman_utils.BoundingBox(box[0] , box[1] , box[2] , box[3] , center = True , gt = gt)
            if gt:
                b_object.add_confidence(1.)
                b_object.add_class(0)
            else:
                b_object.add_confidence(float(box[4]))
                b_object.add_class(int(box[5]))
            box_objects_this_instance.append(b_object)
        box_objects.append(box_objects_this_instance)
    return box_objects

def get_box_objects_per_class(box_objects , num_classes , num_instances):
    obj_per_class = [[] for i in range(num_classes)]

    # box_objects.shape = [batch_size , obj_per_instance]
    # run through each instances , arrange the boxes per class manner
    for i in range(num_instances):
        box_objects_this_instance = box_objects[i]
        for obj in box_objects_this_instance:
            cls_this_object = obj.cls
            if cls_this_object >= num_classes:
                raise Excpetion("Invalid class encountered while calcuating metrics")

            obj_per_class[cls_this_object].append(obj)

    # sort the bbox per class wrt their conf , in decreasing conf manner.
    for c in range(num_classes):
        bbox_this_class = obj_per_class[c]
        if len(bbox_this_class) > 0:
            sorted_bbox_this_class = sorted(bbox_this_class)
            obj_per_class[c] = sorted_bbox_this_class[::-1]
    return obj_per_class
