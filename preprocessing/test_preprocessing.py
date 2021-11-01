import cv2
import sys
import math
import numpy as np
import preprocessing as pre_process

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

def test_preprocess():
    batch_size = 10
    num_batch = 2

    anchors = np.array([[0.17028831 , 0.35888521],
                        [0.05563053 , 0.09101727],
                        [0.11255733 , 0.21961425],
                        [0.0347448  , 0.06395953],
                        [0.32428802 , 0.42267646],
                        [0.47664651 , 0.65827237],
                        [0.21481797 , 0.20969635],
                        [0.07297461 , 0.14739788],
                        [0.11702667 , 0.11145465]])

    is_norm = False
    input_size = 416
    is_augment = True
    base_grid_size = 13
    grid_scales = [1,2,4]
    labels = ["Face" , "Non_Face"]
    anchors = (anchors * input_size).reshape(9 , 2)
    new_sorted_anchors = sort_anchors(anchors)
    train_file = "../data/wider_training_data.pickle"
    image_names = "../data/wider_images_names.pickle"
    data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images/"
    batch_generator = pre_process.DataGenerator(416 ,
                                                base_grid_size,
                                                grid_scales,
                                                new_sorted_anchors,
                                                data_path,
                                                train_file,
                                                image_names,
                                                labels,
                                                is_norm,
                                                is_augment,
                                                batch_size)

    print(batch_generator.num_train_instances)
    for index in range(num_batch):
        batch_images , batch_labels , detector_indexes = batch_generator.load_data_for_test(index)
        yolov1_labels , yolov2_labels , yolov3_labels = batch_labels
        print(yolov1_labels.shape , yolov2_labels.shape , yolov3_labels.shape)
        yolos = [yolov1_labels , yolov2_labels , yolov3_labels]
        print("***********************************")
        for i in range(len(batch_images)):
            print("########################")
            org_image = np.array(batch_images[i])
            detectors = detector_indexes[i]
            object_index = 0
            index = 0
            for yolo_id in range(len(grid_scales)):
                label = yolos[yolo_id][i]
                grid_size = base_grid_size * grid_scales[yolo_id]
                mul_factor = input_size / grid_size
                print(f"grid_size : {grid_size} , mul_factor : {mul_factor}")
                for h in range(grid_size):
                    for w in range(grid_size):
                        for a in range(3):
                            if label[h , w , a , 4] == 1:
                                anchor_index = (yolo_id * 3) + a
                                try:
                                    print(f"org_label : {label[h , w , a]}")
                                    print(f"x : {label[h,w,a,0]} , y : {label[h,w,a,1]}")
                                    if label[h , w , a , 0] >= grid_size or label[h , w , a, 1] >= grid_size:
                                        raise Exception("center dimension are not preoper encoded , " , grid_size)
                                    x_center = int(math.floor(label[h , w , a, 0] * mul_factor))
                                    y_center = int(math.floor(label[h , w , a, 1] * mul_factor))
                                    width = int(math.floor(np.exp(label[h , w, a , 2]) * new_sorted_anchors[anchor_index][0]))
                                    height = int(math.floor(np.exp(label[h , w, a , 3]) * new_sorted_anchors[anchor_index][1]))
                                    print(f"final dimesion : {x_center} , {y_center} , {width} , {height}")
                                    x1 = x_center - int(width/2)
                                    y1 = y_center - int(height/2)
                                    x2 = x1 + width
                                    y2 = y1 + height
                                    classes = "FACE" if label[h , w , a , 5] else "NON_FACE"
                                    print(f"detector_index : {detectors[index]}")
                                    print(classes)
                                    cv2.rectangle(org_image , (x1 , y1) , (x2 , y2) , (255 , 0  ,0) , 1)
                                    object_index += 1
                                    index += 1

                                except Exception as e:
                                    print("Exception : " , e)
                                    cv2.imshow("in except block" , org_image)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                                    print(i , object_index , [x_center , y_center , width  , height])
                                    sys.exit(0)

            cv2.imshow("images" , org_image[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    test_preprocess()
