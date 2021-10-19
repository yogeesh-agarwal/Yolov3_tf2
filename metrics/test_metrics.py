from Yolov3_tf2.metrics.unit_metrics import UnitMetrics
from Yolov3_tf2.metrics import precision_recall
from Yolov3_tf2.metrics import utils as m_utils
from Yolov3_tf2.metrics import mAP
from Yolov3_tf2 import utils
import tensorflow as tf
import numpy as np


def test():
    y_true = np.array([utils.BoundingBox(0 , 0, 100 , 200 , gt = True),
                       utils.BoundingBox(150 , 200 , 50 , 50 , gt = True),
                       utils.BoundingBox(0 , 0, 50 , 25 , gt = True)]).reshape(3,)
    for i in range(3):
        y_true[i].add_class(0)
        y_true[i].add_confidence(1)
        print(y_true[i])

    y_pred = np.array([utils.BoundingBox(0 , 0, 110 , 190),
                        utils.BoundingBox(130 , 220, 45 , 55),
                        utils.BoundingBox(400 , 200, 120 , 99)]).reshape(3,)

    for i in range(3):
        y_pred[i].add_class(0)
        y_pred[i].add_confidence(np.random.uniform(0,1))
        print(y_pred[i])

    um = UnitMetrics(0.5 , 1)
    # um.update_state(y_true , y_pred)
    # dictn = um.result()
    # for k in dictn:
    #     print(k , dictn[k].numpy())

    print("****************************************")
    precision = precision_recall.PrecisionOD(um , 1)
    recall = precision_recall.RecallOD(um , 1)
    precision.update_state(y_true , y_pred)
    recall.update_state(y_true , y_pred)
    print("precision_array : " , precision.result().numpy() , " , precision : " , precision.result().numpy()[-1])
    print("recall_array : " , recall.result().numpy() , " , recall : " , recall.result().numpy()[-1])


    print("****************************************")
    f1 = mAP.F1_score(precision, recall , 1)
    f1.update_state()
    print("F1_Score : " , f1.result().numpy())

    print("****************************************")
    map = mAP.AveragePrecision(precision , recall , 1)
    map.update_state()
    print("mAP : " , map.result().numpy())

test()
