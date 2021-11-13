import tensorflow as tf
import numpy as np

# calculate the precision of a class.
class PrecisionOD(tf.keras.metrics.Metric):
    def  __init__(self , unit_metrics , num_classes , name = "Precision" , **kwargs):
        super().__init__(name = name , **kwargs)
        self.epsilon = 1e-5
        self.num_classes = num_classes
        self.unit_metrics = unit_metrics
        # array to calculate the AP metric for a single batch using this precision array
        self.precision = tf.Variable(0 ,
                                     shape = tf.TensorShape(None) ,
                                     validate_shape = False ,
                                     dtype = tf.float32,
                                     name = "precision")

    def update_state(self , gt_box_objects , pred_box_objects , sample_weights = None):
        # get the latest states of true_pos , false_pos , false_neg for precision calculations
        # precision = tp / (tp + fp)
        self.unit_metrics.update_state(gt_box_objects , pred_box_objects)
        unit_metrics_state = self.unit_metrics.result()
        tp = unit_metrics_state["true_pos"].numpy()
        fp = unit_metrics_state["false_pos"].numpy()
        precision = tp / (tp + fp + self.epsilon)
        self.precision.assign(precision)

    # return the precision array
    def result(self):
        return self.precision

    def reset_state(self):
        self.precision.assign(0)

# calculates the recall of a class.
class RecallOD(tf.keras.metrics.Metric):
    def  __init__(self , unit_metrics , num_classes , name = "Recall" , **kwargs):
        super().__init__(name = name , **kwargs)
        self.num_classes = num_classes
        self.unit_metrics = unit_metrics
        self.epsilon = 1e-5
        # array to calculate the AP metric using this recall array.
        self.recall = tf.Variable(0 ,
                                  shape = tf.TensorShape(None) ,
                                  validate_shape = False ,
                                  dtype = tf.float32,
                                  name = "recall")

    def update_state(self , gt_box_objects, pred_box_objects , sample_weights = None):
        # get the latest states of true_pos , false_pos , false_neg for recall calculations
        # recall = tp / (tp + fn)
        # keep in mind to call recall only afrer precision is called, cause we are calling unitmetrics update_state only once.
        unit_metrics_state = self.unit_metrics.result()
        tp = unit_metrics_state["true_pos"].numpy()
        num_gt_objects = len(gt_box_objects)
        recall = tp / (num_gt_objects + self.epsilon)
        self.recall.assign(recall)

    # return recall array for AP calculation
    def result(self):
        return self.recall

    def reset_state(self):
        self.recall.assign(0)
