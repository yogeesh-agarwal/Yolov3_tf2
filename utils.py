import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class BoundingBox:
    def __init__(self , x , y , w , h , center = False , gt = False):
        self.x = x # top_left_x
        self.y = y # top_left_y
        self.w = w
        self.h = h
        if center:
            # if center coord , convert to top_left coord system.
            self.x = self.x - self.w*0.5
            self.y = self.y - self.h*0.5
        self.ground_truth = gt
        self.matched = False
        self.iou = 0

    def add_class(self , cls):
        self.cls = cls

    def add_confidence(self , conf):
        self.conf = conf

    def set_status(self , status):
        self.matched = status

    def set_matched_box(self , box):
        self.matched_box = box

    def get_area(self):
        return self.w * self.h

    def reset(self):
        self.iou = 0
        self.matched = False
        self.matched_box = None

    def convert2xyxy(self):
        x2 = self.x + self.w
        y2 = self.y + self.h
        return np.array([self.x , self.y , x2 , y2])

    def cal_iou(self , other):
        transformed_box1 = self.convert2xyxy()
        transformed_box2 = other.convert2xyxy()
        overlap_mins = []
        overlap_mins.append(np.maximum(transformed_box1[0] , transformed_box2[0]))
        overlap_mins.append(np.maximum(transformed_box1[1] , transformed_box2[1]))
        overlap_maxs = []
        overlap_maxs.append(np.minimum(transformed_box1[2] , transformed_box2[2]))
        overlap_maxs.append(np.minimum(transformed_box1[3] , transformed_box2[3]))
        intersect_wh = np.maximum(np.array(overlap_maxs) - np.array(overlap_mins) , 0.)

        intersect_area = intersect_wh[... , 0] * intersect_wh[... , 1]
        union_area = self.get_area() + other.get_area() - intersect_area
        return float(intersect_area) / union_area

    def match_detections(self , detection_boxes , threshold):
        for det_box in detection_boxes:
            if det_box.cls != self.cls:
                raise Exception("Class mismatch found")

            iou_this_box = self.cal_iou(det_box)
            if iou_this_box > self.iou:
                self.iou = iou_this_box

            if iou_this_box >= threshold and not det_box.matched:
                det_box.set_status(True)
                self.matched = True
                self.matched_box = det_box
                det_box.set_matched_box(self)
                return True

        return False

    def print_matched_box(self):
        if self.matched:
            return "\n{} , {} , {} , {}".format(self.matched_box.x , self.matched_box.y , self.matched_box.w , self.matched_box.h)
        else:
            return "\nNO matched box found for this Bounding_box"

    def save_bbox(self , file):
        coords = self.convert2xyxy()
        if self.ground_truth:
            file.write("FACE {} {} {} {}\n".format(int(coords[0]) , int(coords[1]) , int(coords[2]) , int(coords[3])))
        else:
            file.write("FACE {} {} {} {} {}\n".format(self.conf , int(coords[0]) , int(coords[1]) , int(coords[2]) , int(coords[3])))

    def __repr__(self):
        return "center_X = {} , center_Y = {} , width = {} , height = {} , class = {} , iou = {} , conf = {}".format(self.x , self.y , self.w , self.h , self.cls , self.iou ,self.conf)

    def __eq__(self , other):
        return self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h and self.conf == other.conf and self.cls == other.cls

    def __lt__(self, other):
        return self.conf < other.conf

    def __gt__(self , other):
        return self.conf > other.conf

    def __ge__(self , other):
        return self.conf >= other.conf

def gen_cell_grid(grid_w , grid_h , num_anchors):
    cell_grid = np.zeros((grid_h , grid_w , num_anchors , 2) , dtype = np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            for anc_index in range(num_anchors):
                cell_grid[row , col , anc_index , 0] = col
                cell_grid[row , col , anc_index , 1] = row

    return cell_grid

def cal_iou(gt_xy , gt_wh , pred_xy , pred_wh):
    gt_mins = gt_xy - (gt_wh / 2.)
    gt_maxs = gt_xy + (gt_wh / 2.)
    pred_mins = pred_xy - (pred_wh / 2.)
    pred_maxs = pred_xy + (pred_wh / 2.)

    overlap_mins = tf.maximum(gt_mins , pred_mins)
    overlap_maxs = tf.minimum(gt_maxs , pred_maxs)
    overlap_wh = tf.maximum(overlap_maxs - overlap_mins , 0.)

    gt_area = gt_wh[... , 0] * gt_wh[... , 1]
    pred_area = pred_wh[... , 0] * pred_wh[... , 1]
    overlap_area = overlap_wh[... , 0] * overlap_wh[... , 1]
    union_area = pred_area + gt_area - overlap_area
    iou = tf.truediv(overlap_area , union_area)
    return iou

def cal_giou(gt_xy , gt_wh , pred_xy , pred_wh):
    gt_mins = gt_xy - (gt_wh / 2.)
    gt_maxs = gt_xy + (gt_wh / 2.)
    pred_mins = pred_xy - (pred_wh / 2.)
    pred_maxs = pred_xy + (pred_wh / 2.)

    overlap_mins = tf.maximum(gt_mins , pred_mins)
    overlap_maxs = tf.minimum(gt_maxs , pred_maxs)
    overlap_wh = tf.maximum(overlap_maxs - overlap_mins , 0.)

    gt_area = gt_wh[... , 0] * gt_wh[... , 1]
    pred_area = pred_wh[... , 0] * pred_wh[... , 1]
    overlap_area = overlap_wh[... , 0] * overlap_wh[... , 1]
    union_area = pred_area + gt_area - overlap_area
    iou = tf.truediv(overlap_area , union_area)

    # calculate smallest closed convex surface:
    convex_mins = tf.minimum(gt_mins , pred_mins)
    convex_maxs = tf.maximum(gt_maxs , pred_maxs)
    convex_wh = tf.maximum(convex_maxs - convex_mins , 0.)
    convex_area = convex_wh[... , 0] * convex_wh[... , 1]
    giou = iou - ((convex_area - union_area) / (convex_area + 1e-5))
    return tf.expand_dims(tf.expand_dims(giou , axis = 0) , axis = -1)


def sort_anchors(anchors):
    anchors = np.reshape(anchors, [9,2])
    anchor_areas = {}
    for anchor in anchors:
        area = anchor[0]*anchor[1]
        anchor_areas[area] = anchor

    sorted_areas = sorted(list(anchor_areas.keys()))
    sorted_anchors = []
    for area in sorted_areas[::-1]:
        print(anchor_areas[area] , area)
        sorted_anchors.append(anchor_areas[area])

    return np.array(sorted_anchors).reshape([9,2])

def shuffle_array(shuffled_index_list , org_array):
    assert len(shuffled_index_list) == len(org_array)
    shuffled_array = np.empty(org_array.shape , dtype = org_array.dtype)
    print("***********************************************")
    for index , shuffled_index in enumerate(shuffled_index_list):
        print(index , shuffled_index)
        shuffled_array[index] = org_array[shuffled_index]

    return shuffled_array


def convert_to_file(ground_truth , detection , gt_file , det_file):
    with open(gt_file , "w") as file:
        for object in ground_truth:
            object.save_bbox(file)

    with open(det_file , "w") as file:
        for object in detection:
            object.save_bbox(file)
