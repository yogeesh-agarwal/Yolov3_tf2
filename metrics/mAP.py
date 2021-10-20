from Yolov3_tf2.postprocessing import post_processing
from Yolov3_tf2.metrics import utils as metric_utils
from Yolov3_tf2.metrics import unit_metrics
from Yolov3_tf2.metrics import precision_recall
import tensorflow as tf
import numpy as np

# calculates f1_score for a class
class F1_score(tf.keras.metrics.Metric):
    def __init__(self , precision , recall , num_classes , name = "F1_score" , **kwargs):
        super().__init__(name = name , **kwargs)
        self.recall = recall
        self.precision = precision
        self.num_classes = num_classes
        # mean metric to track f1_score across all batches.
        self.f1_score = tf.Variable(0  ,name = "f1_score" , dtype = tf.float32)

    def update_state(self , y_true = None , y_pred = None , sample_weights = None):
        curr_precision = self.precision.result().numpy()
        curr_recall = self.recall.result().numpy()
        f1 = (2 * curr_precision * curr_recall) / (curr_precision + curr_recall)
        self.f1_score.assign(f1[-1])

    def result(self):
        return self.f1_score

    def reset_state(self):
        self.f1_score.assign(0)

# calculates the "average_precision" for a single class.
class AveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, precision , recall , num_classes , name = "average_precision" , **kwargs):
        super().__init__(name = name , **kwargs)
        self.recall = recall
        self.precision = precision
        self.num_classes = num_classes
        # mean metric to track AP across all batches
        self.average_precision = tf.Variable(0. , name = "average_precision" , dtype = tf.float32)

    """
    call to precision and recall metric update_state functions is kept independent deliberately.
    this is to ensure that there is no dependency of calls in this class and to reduce calls to unit metric class.
    So one can assume that precision and recall update_states are already called prior to this fn call.
    """
    def update_state(self , y_true = None , y_pred = None , sample_weights = None):
        prec_array = self.precision.result()
        rec_array = self.recall.result()
        # append 0's in start and end  for prec_array for calculation below (Y axis)
        prec_array = np.concatenate([np.array([0.]) , prec_array , np.array([0.])] , axis = -1)
        for i in range(prec_array.shape[0] - 1 , 0 , -1):
            prec_array[i-1] = max(prec_array[i-1] , prec_array[i])

        # append 0 in start and 1 in end , as recall will be on X axis.
        rec_array = np.concatenate([np.array([0.]) , rec_array , np.array([1.])] , axis = -1)
        rec_drops_idx = []
        for i in range(rec_array.shape[0] - 1):
            if rec_array[i+1] != rec_array[i]:
                rec_drops_idx.append(i+1)

        # we calculate ap by simplifying the PR curve into rectangles , thus
        # area under rectangles (W*H) = difference of X axis (W) * precision(H) for that delta recall.
        # finally sum all the rectangles areas.
        ap = 0
        for i in rec_drops_idx:
            ap += np.sum((rec_array[i] - rec_array[i-1]) * prec_array[i])

        # calculate mean AP for "this class" for every batch.
        self.average_precision.assign(ap)

    def result(self):
        return self.average_precision

    def reset_state(self):
        return self.average_precision.assign(0.)

# calculates mAP for all classes , across all batches.
class MeanAveragePrecision(tf.keras.metrics.Metric):
    def __init__(self ,
                 num_classes ,
                 anchors ,
                 threshold = 0.5 ,
                 name = "mAP" ,
                 **kwargs):
        super().__init__(name = name , **kwargs)
        self.anchors = anchors
        self.num_classes = num_classes
        self.unit_metrics = unit_metrics.UnitMetrics(threshold , self.num_classes)
        self.recall = precision_recall.RecallOD(self.unit_metrics , self.num_classes)
        self.precision = precision_recall.PrecisionOD(self.unit_metrics , self.num_classes)
        self.average_precision = AveragePrecision(self.precision , self.recall , self.num_classes)

        # metric to keep track of AP for all classes , mean of this metric will be input to mAP metric.
        self.ap_per_class = tf.Variable(np.zeros(shape = [self.num_classes], dtype = np.float32) ,
                                        shape = (self.num_classes,) ,
                                        dtype = tf.float32 ,
                                        name = "AP_per_class")
        # metric to keep track of mAP for all classes across all batches for an epoch
        self.mAP = tf.keras.metrics.Mean(name = "mAP")


    def update_state(self , ground_truth , predictions , sample_weights = None):
        box_objects = post_processing.post_process(predictions ,
                                                   self.anchors ,
                                                   ground_truth = ground_truth ,
                                                   num_classes = self.num_classes)
        num_instances = pred_box_objects.shape[0]
        #shape = [batch_size , objects_per_instance]
        pred_box_objects , gt_box_objects = box_objects

        # shape = [num_class , objects_per_class]
        gt_boxes_per_class = metric_utils.get_boxes_per_class(gt_box_objects, self.num_classes , num_instances)
        pred_boxes_per_class = metric_utils.get_boxes_per_class(pred_box_objects, self.num_classes , num_instances)
        ap_per_class = np.zeros(shape = (self.num_classes) , dtype = np.float32)
        for c in range(num_classes):
            gt_boxes_this_cls = gt_boxes_per_class[c]
            pred_boxes_this_cls = pred_boxes_per_class[c]
            self.precision.update_state(gt_boxes_this_cls, pred_boxes_this_cls)
            self.recall.update_state(gt_boxes_this_cls , pred_boxes_this_cls)
            self.average_precision.update_state()
            ap_per_class[c] = self.average_precision.result().numpy()

        # assign ap_per_class metric , for this batch
        self.ap_per_class.assign(ap_per_class)
        mAP = tf.reduce_mean(ap_per_class , axis = -1)
        self.mAP.update_state(mAP)

    def result(self):
        return self.mAP.result()

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        self.average_precision.reset_state()
        self.ap_per_class = tf.zeros(shape = (self.num_classes,))
        self.mAP.reset_state()
