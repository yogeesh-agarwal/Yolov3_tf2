from Yolov3_tf2 import utils
import tensorflow as tf
import numpy as np
import sys

class Yolov3Loss(tf.keras.losses.Loss):
    def __init__(self , base_grid , grid_scale , num_classes , anchors , name = "Yolov3_loss"):
        super().__init__(name = name)
        self.epsilon = 1e-6
        self.input_dim = 416
        self.obj_scale = 1.0
        self.noobj_scale = 1.0
        self.class_scale = 1.0
        self.coord_scale = 5.0
        self.ignore_thresh = 0.5
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.grid_dim = base_grid * grid_scale
        self.anchors = tf.reshape(anchors , [1,1,1,3,2])
        if grid_scale == 1:
            self.mask_logging_string = "file://../data/large_objects_masks.txt"
            self.logging_string = "file://../data/large_object_losses.txt"
        elif grid_scale == 2:
            self.mask_logging_string = "file://../data/medium_objects_masks.txt"
            self.logging_string = "file://../data/medium_object_losses.txt"
        else:
            self.mask_logging_string = "file://../data/small_objects_masks.txt"
            self.logging_string = "file://../data/small_object_losses.txt"
        self.grid_norm = tf.reshape(tf.cast([self.grid_dim , self.grid_dim] , tf.float32) , [1,1,1,1,2])
        self.net_norm = tf.reshape(tf.cast([self.input_dim , self.input_dim] , tf.float32) , [1,1,1,1,2])

    def modify_predictions(self):
        # shape of predictions are : [batch_size , grid_dim , grid_dim , num_anchors , 4 + 1 + num_classes]
        cell_grid = utils.gen_cell_grid(self.grid_dim , self.grid_dim , self.num_anchors)
        pred_xy = tf.sigmoid(self.predictions[... , 0:2]) + cell_grid
        pred_wh = self.predictions[... , 2:4]
        pred_conf = tf.sigmoid(self.predictions[... , 4])
        pred_class = self.predictions[... , 5:]
        return pred_xy , pred_wh , pred_conf , pred_class

    def modify_gt(self):
        gt_xy = self.ground_truth[... , 0:2]
        gt_wh = self.ground_truth[... , 2:4]
        gt_conf = self.ground_truth[... , 4]
        gt_class = tf.argmax(self.ground_truth[... , 5:] , -1)
        return gt_xy , gt_wh , gt_conf , gt_class

    def gen_masks(self, gt_xy , gt_wh , gt_conf , pred_conf , pred_xy , pred_wh):
        #coord_mask
        coord_mask = gt_conf
        weight_scale = tf.exp(gt_wh) * self.anchors / self.net_norm
        weight_scale = 2 - weight_scale[... , 0] * weight_scale[... , 1]
        coord_mask = tf.expand_dims(coord_mask * weight_scale * self.coord_scale , 4)

        #object_mask / no_object_mask
        object_mask = gt_conf
        no_object_mask = (1 - object_mask) * self.noobj_scale
        object_mask *= self.obj_scale

        #ignore_mask
        ignore_mask = pred_conf - 0
        pred_gt_iou = utils.cal_iou(gt_xy / self.grid_norm , tf.exp(gt_wh) * self.anchors / self.net_norm ,
                                    pred_xy / self.grid_norm , tf.exp(pred_wh) * self.anchors / self.net_norm)
        best_overlaps = tf.reduce_max(pred_gt_iou)
        ignore_mask = ignore_mask * tf.cast(best_overlaps < self.ignore_thresh , tf.float32)

        #class_mask
        class_mask = gt_conf * self.class_scale
        return coord_mask , object_mask , no_object_mask , ignore_mask , class_mask

    def localization_loss(self, coord_mask , gt , preds):
        gt_xy , gt_wh = gt
        pred_xy , pred_wh = preds
        xy_loss = tf.reduce_sum(tf.square(coord_mask * (pred_xy - gt_xy)))
        wh_loss = tf.reduce_sum(tf.square(coord_mask * (pred_wh - gt_wh)))
        return xy_loss + wh_loss

    def conf_loss(self , conf_masks , gt_conf , pred_conf):
        object_mask , no_object_mask , ignore_mask = conf_masks
        object_del = object_mask * (pred_conf - gt_conf)
        noobject_del = no_object_mask * ignore_mask
        conf_loss = tf.reduce_sum(tf.square(object_del + noobject_del))
        return conf_loss

    def class_loss(self , class_mask , gt_class , pred_class):
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gt_class , logits = pred_class)
        class_loss = tf.reduce_sum(class_loss * class_mask)
        return class_loss

    def get_loss(self, ground_truth , predictions):
        self.predictions = predictions
        self.ground_truth = ground_truth
        gt_xy , gt_wh , gt_conf , gt_class = self.modify_gt()
        pred_xy , pred_wh , pred_conf , pred_class = self.modify_predictions()
        coord_mask , object_mask , no_object_mask , ignore_mask , class_mask = self.gen_masks( gt_xy , gt_wh , gt_conf , pred_conf , pred_xy , pred_wh)

        coord_loss = self.localization_loss(coord_mask ,
                                            [gt_xy , gt_wh] ,
                                            [pred_xy , pred_wh])

        conf_loss =  self.conf_loss([object_mask , no_object_mask , ignore_mask] ,
                                    gt_conf ,
                                    pred_conf)
        class_loss = self.class_loss(class_mask ,
                                     gt_class,
                                     pred_class)
        loss = (coord_loss + conf_loss + class_loss) * self.grid_dim  # multiply with grid dim as current loss is normalized , cal wrt to the grid.
        tf.print("coord_loss : " , coord_loss , ", conf_loss : " , conf_loss , ", class_loss : " , class_loss)
        return loss

    # overriding call method to implement custom loss logic
    def call(self , ground_truth , predictions):
        loss = self.get_loss(ground_truth, predictions)
        return loss
