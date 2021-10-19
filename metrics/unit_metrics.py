from Yolov3_tf2.metrics import utils as metric_utils
import tensorflow as tf
import numpy as np

# unitmetric class to calculate the true_positive , false_pos of a batch for a single class given its detections(preds) and annotations(gt).
class UnitMetrics(tf.keras.metrics.Metric):
    def __init__(self , threshold , num_classes , name = "TP_FP" , **kwargs):
        super().__init__(name = name , **kwargs)
        self.threshold = threshold
        self.num_classes = num_classes
        # create vars to track the metric
        self.tp = tf.Variable(0 ,
                              shape = tf.TensorShape(None) ,
                              name = "true_pos" ,
                              validate_shape = False ,
                              dtype = tf.float32)
        self.fp = tf.Variable(0 ,
                              shape = tf.TensorShape(None) ,
                              name = "false_pos" ,
                              validate_shape = False ,
                              dtype = tf.float32)

    def update_state(self, gt_box_objects , pred_box_objects , sample_weights = None):
        # box_objects are objects of BoundingBox class ,shape : [objects_this_class]
        tp = []
        fp = []
        # if there are gt for this clss:
        if len(gt_box_objects) > 0:
            # for obj in predictions for single class
            for obj in pred_box_objects:
                match_found = obj.match_detections(gt_box_objects , self.threshold)
                # if match is found , that is iou > threshold
                if match_found:
                    tp.append(1)
                    fp.append(0)

                # else this ground truth is gone undetected.
                else:
                    tp.append(0)
                    fp.append(1)

        # else every prediction is a FP for this class.
        elif len(pred_box_objects) > 0:
                tp = [0 for i in range(len(pred_box_objects))]
                fp = [1 for i in range(len(pred_box_objects))]
        # assign the TP/FP stats for this batch so tracked_vars.
        self.tp.assign(tp)
        self.fp.assign(fp)

    def result(self):
        # return the dictionary of cumulative sum of tp / fp
        outputs = {}
        outputs["true_pos"] = tf.math.cumsum(self.tp)
        outputs["false_pos"] = tf.math.cumsum(self.fp)
        return outputs

    def reset_states(self):
        # assign 0 to tp/fp
        self.tp.assign(0)
        self.fp.assign(0)
