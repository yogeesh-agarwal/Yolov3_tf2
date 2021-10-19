import os
import sys
import cv2
import math
import pickle
import numpy as np

def cal_overlap_ratio(box_w , box_h , image_height ,image_width , threshold = 0.1):
    image_area = image_height * image_width
    box_area = box_h * box_w

    if  box_area / image_area < threshold:
        print("discarding image")
        return False
    print(image_area , box_area , box_area / image_area)
    return True

def read_files(filepath , data_dir , prune_dataset = False):
    file = open(filepath , "r")
    lines = file.readlines()
    train_data = {}
    image_names = []
    count = 1
    num_faces = [str(i)+"\n" for i in range(30)]
    try:
        while count < len(lines):
            valid_file = False
            length = int(lines[count].split("\n")[0])
            if lines[count] in num_faces:
                if lines[count] == "0\n":
                    print("0 faces in the image")
                    raise Exception("Zero faces in the image , " , lines[count-1])

                valid_file = True
                filename = lines[count-1][:-1]
                event_class =  int(filename.split("--")[0])
                image = cv2.imread(os.path.join(data_dir, filename) , 1)

                if length == 0:
                    count += 3
                    continue
                count += 1
                boxes = []
                discard_image = False
                for bb in range(length):
                    box = lines[count + bb].strip()
                    box = list(map(int , box.split(" ")))
                    if prune_dataset:
                        if not cal_overlap_ratio(box[2] , box[3] , image.shape[0] , image.shape[1] , 0.001):
                            discard_image = True
                            break
                    bbox = box[:4]
                    boxes.append(bbox)
                if not discard_image:
                    image_path = os.path.join(data_dir , filename)
                    train_data[filename] = boxes
                    image_names.append(filename)
            count += (length + 1)
            if not valid_file:
                count += 1
    except Exception as e:
        print("error encontered : " , e)
        print(count , lines[count])
        print(filename , event_class)
        sys.exit(0)

    return train_data , image_names

def normalize_coords(train_data , data_path):
    for index , img_name in enumerate(train_data):
        image_path = os.path.join(data_path , img_name)
        objects = train_data[img_name]
        image = cv2.imread(image_path , 1)
        height , width = image.shape[0] , image.shape[1]
        norm_objects = []
        for object in objects:
            object[0] /= width
            object[1] /= height
            object[2] /= width
            object[3] /= height
            norm_objects.append(object)
        print("{} image objects are normalized with height = {} , width = {}".format(index+1 , height , width))
        train_data[img_name] = norm_objects
    return train_data

def test_image(data_path , filename , objects):
    image_path = os.path.join(data_path, filename)
    image = cv2.imread(image_path , 1)
    for object in objects:
        x1 = object[0]
        y1 = object[1]
        x2 = x1 + object[2]
        y2 = y1 + object[3]
        cv2.rectangle(image, (x1,y1) , (x2,y2) , (255 , 0 , 0) , 3)

    cv2.imshow("in_testing_module" , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(datapath , filename , h , w , norm_objects):
    image_path = os.path.join(datapath , filename)
    image = cv2.imread(image_path , 1)
    org_h , org_w = image.shape[0] , image.shape[1]

    resized_objects = []
    for object in norm_objects:
        new_object = []
        new_object.append(object[0] * w)
        new_object.append(object[1] * h)
        new_object.append(object[2] * w)
        new_object.append(object[3] * h)
        resized_objects.append(new_object)

    resized_image = cv2.resize(image , (w , h))
    for object in resized_objects:
        x1 = int(object[0])
        y1 = int(object[1])
        x2 = int(x1 + object[2])
        y2 = int(y1 + object[3])
        cv2.rectangle(resized_image, (x1,y1) , (x2,y2) , (255 , 0 , 0) , 3)

    cv2.imshow("in_resize_module" , resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train_data , image_names = read_files("/home/yogeesh/yogeesh/datasets/face_detection/wider face/wider_face_split/wider_face_split/wider_face_train_bbx_gt.txt" , "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images" , prune_dataset = True)
    for i in range(10):
        test_image("/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images" , list(train_data.keys())[i] , train_data[list(train_data.keys())[i]])
    train_data = normalize_coords(train_data , "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images")
    for i in range(10):
        resize_image("/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images" , list(train_data.keys())[i] , 416 , 416 , train_data[list(train_data.keys())[i]])
    for index , img_path in enumerate(train_data):
        print(img_path)
        for object in train_data[img_path]:
            print(f"************{object}****************")

    with open("../data/wider_training_data.pickle" , "wb") as f:
        pickle.dump(train_data , f , pickle.HIGHEST_PROTOCOL)
    with open("../data/wider_images_names.pickle" , "wb") as f:
        pickle.dump(image_names , f , pickle.HIGHEST_PROTOCOL)

    print("training data and image_names are stored in piclke file.")

    print("******************************************************")

    val_data , val_images_names = read_files("/home/yogeesh/yogeesh/datasets/face_detection/wider face/wider_face_split/wider_face_split/wider_face_val_bbx_gt.txt" , "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_val/WIDER_val/images/" , prune_dataset = True)
    val_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_val/WIDER_val/images"
    for i in range(10):
        test_image(val_data_path , list(val_data.keys())[i] , val_data[list(val_data.keys())[i]])
    val_data = normalize_coords(val_data , val_data_path)
    for i in range(10):
        resize_image(val_data_path , list(val_data.keys())[i] , 416 , 416 , val_data[list(val_data.keys())[i]])
    for index , img_path in enumerate(val_data):
        print(img_path)
        for object in val_data[img_path]:
            print(object)

    with open("../data/wider_validation_data.pickle" , "wb") as f:
        pickle.dump(val_data , f , pickle.HIGHEST_PROTOCOL)
    with open("../data/wider_val_images_names.pickle" , "wb") as f:
        pickle.dump(val_images_names , f , pickle.HIGHEST_PROTOCOL)

    print("validation data and image_names are stored in piclke file.")
