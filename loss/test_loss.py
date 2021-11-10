import tensorflow as tf
import numpy as np
import loss as loss

def test_loss():
    gt_yolo1 = np.random.random(size = [1,13,13,3,7]).astype(np.float32)
    gt_yolo2 = np.random.random(size = [1,26,26,3,7]).astype(np.float32)
    gt_yolo3 = np.random.random(size = [1,52,52,3,7]).astype(np.float32)
    pred_yolo_1 = np.random.random(size = [1,13,13,3,7]).astype(np.float32)
    pred_yolo_2 = np.random.random(size = [1,26,26,3,7]).astype(np.float32)
    pred_yolo_3 = np.random.random(size = [1,52,52,3,7]).astype(np.float32)
    anchors = np.random.random(size = [9,2]).astype(np.float32)

    yolo_loss1 = loss.Yolov3Loss(13 , 1 , 2 , anchors[:3])
    yolo_loss2 = loss.Yolov3Loss(26, 1 , 2 , anchors[3:6])
    yolo_loss3 = loss.Yolov3Loss(52, 1 , 2 , anchors[6:])

    loss1 = yolo_loss1(gt_yolo1 , pred_yolo_1)
    loss2 = yolo_loss2(gt_yolo2 , pred_yolo_2)
    loss3 = yolo_loss3(gt_yolo3 , pred_yolo_3)
    print(loss1 , loss2 , loss3)

if __name__ == "__main__":
    test_loss()
