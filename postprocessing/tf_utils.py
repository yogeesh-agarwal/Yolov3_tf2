import os
import cv2
import numpy as np
import tensorflow as tf

class TFBoundingBox(tf.experimental.ExtensionType):
    x : tf.Tensor
    y : tf.Tensor
    w : tf.Tensor
    h : tf.Tensor
    iou : tf.Tensor
    cls : tf.Tensor
    conf : tf.Tensor
    matched : tf.Tensor
    def __init__(self , x , y , w , h ,
                 conf ,
                 cls ,
                 matched ,
                 center = False ,
                 gt = False):

        self.x = x # top_left_x
        self.y = y # top_left_y
        self.w = w
        self.h = h
        self.iou = 0
        self.cls = cls
        self.conf = conf
        self.matched = matched
        if center:
            # if center coord , convert to top_left coord system.
            self.x = self.x - self.w*0.5
            self.y = self.y - self.h*0.5

    def add_class(self , cls):
        return (self.x , self.y , self.w , self.h , self.conf , cls , self.matched)

    def add_confidence(self , conf):
        return (self.x , self.y , self.w , self.h , conf , self.cls , self.matched)

    def set_status(self , status):
        return (self.x , self.y , self.w , self.h , self.conf , self.cls , status)

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

def sigmoid(x):
    return tf.math.sigmoid(x)

def softmax(x , axis = -1 , t = -100):
    return tf.nn.softmax(x)

def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def modify_locs_util(localizations , anchors , img_shape = [416, 416] , ground_truth = False):
    locs_shape = tf.shape(localizations)
    batch_size = locs_shape[0]
    grid_shape = locs_shape[1:3]
    num_anchors = locs_shape[3]
    num_classes = locs_shape[4] - 5
    strides = [img_shape[0] // grid_shape[0], img_shape[1] // grid_shape[1]]
    cell_grid = gen_cell_grid(grid_shape[0] , grid_shape[1] , batch_size)

    if not ground_truth:
        conf = sigmoid(localizations[... , 4])
        classes = softmax(localizations[... , 5:])
        xy = (sigmoid(localizations[... , 0:2]) + cell_grid) * strides
        wh = tf.exp(tf.cast(localizations[... , 2:4] , dtype = tf.float32)) * anchors * strides

    else:
        xy = localizations[... , 0:2]
        wh = localizations[... , 2:4]
        conf = localizations[... , 4]
        classes = localizations[... , 5:]

    xy = tf.cast(xy , dtype = tf.float32)
    conf = tf.cast(conf , dtype = tf.float32)
    classes = tf.cast(classes , dtype = tf.float32)
    xy = tf.cast(tf.reshape(xy , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 2]) , dtype = tf.float32)
    wh = tf.cast(tf.reshape(wh , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 2]) , dtype = tf.float32)
    conf = tf.cast(tf.reshape(conf , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 1]) , dtype = tf.float32)
    classes = tf.cast(tf.reshape(classes , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 2]) , dtype = tf.float32)
    modified_locs = tf.concat([xy , wh , conf , classes] , axis = -1)
    return tf.reshape(modified_locs , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 5+num_classes])

def iou(boxes , box):
    boxes_x_min = boxes[... , 0] - boxes[... , 2]*0.5
    boxes_y_min = boxes[... , 1] - boxes[... , 3]*0.5
    boxes_x_max = boxes[... , 0] + boxes[... , 2]*0.5
    boxes_y_max = boxes[... , 1] + boxes[... , 3]*0.5

    ref_x_min = box[0] - box[2]*0.5
    ref_y_min = box[1] - box[3]*0.5
    ref_x_max = box[0] + box[2]*0.5
    ref_y_max = box[1] + box[3]*0.5

    intersected_width = tf.maximum(tf.minimum(boxes_x_max , ref_x_max) - tf.maximum(boxes_x_min , ref_x_min) , 0)
    intersected_height = tf.maximum(tf.minimum(boxes_y_max , ref_y_max) - tf.maximum(boxes_y_min , ref_y_min) , 0)
    intersection = intersected_width * intersected_height
    union = (boxes[... , 2] * boxes[... , 3]) + (box[... , 2] * box[... , 3]) - intersection
    return intersection / union

def gen_cell_grid(grid_h , grid_w , batch_size):
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)) , tf.float32)
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.cast(tf.tile(tf.concat([cell_x,cell_y],-1) , [batch_size , 1 , 1 , 3 , 1]) , dtype = tf.float32)
    return cell_grid

def read_tf_file(filename):
    raw = tf.io.read_file(filename)
    tf.print(raw)
    return raw

def draw_predictions(img , preds , class_names):
    for i in range(preds.shape[0]):
        curr_box = preds[i][:4]
        curr_prob = float(preds[i][4])
        xmin = int(curr_box[0] - curr_box[2]*0.5)
        ymin = int(curr_box[1] - curr_box[3]*0.5)
        xmax = int(curr_box[0] + curr_box[2]*0.5)
        ymax = int(curr_box[1] + curr_box[3]*0.5)
        conf = str(tf.round(curr_prob , 2).numpy() * 100)
        pred_class = int(preds[i][5])
        label = class_names[pred_class]
        cv2.rectangle(img , (xmin , ymin) , (xmax , ymax) , (0 , 0 , 255) , 2)
        cv2.putText(img, label + " " + conf + "%" , (xmin , ymin) , cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 0, 0), 2)

    cv2.imshow("inference_img", img[:,:,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
