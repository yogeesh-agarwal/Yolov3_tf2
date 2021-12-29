import cv2
import time
import numpy as np
from Yolov3_tf2.postprocessing import tf_utils as pp_utils
from Yolov3_tf2.postprocessing import tf_postprocessing as post_processing

class TestingModules():
    def __init__(self ,
                 num_batch ,
                 input_size,
                 sorted_anchors,
                 class_names):
        self.num_batch = num_batch
        self.input_size = input_size
        self.class_names = class_names
        self.sorted_anchors = sorted_anchors

    def predict_webcam_input(self , model):
        if model is None:
            raise Exception("Detection model cant be none")

        cap = cv2.VideoCapture(0)
        while(cap.isOpened()):
            stime = time.time()
            ret , frame = cap.read()
            preprocessed_frame = np.array(cv2.resize(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB) , (self.input_size,self.input_size)) , dtype = np.float32)
            inp_frame = preprocessed_frame.reshape(1,self.input_size,self.input_size,3) / 255.0
            predictions_dict = model(inp_frame , training = False)
            large_scale_preds = predictions_dict["large_scale_preds"]
            medium_scale_preds = predictions_dict["medium_scale_preds"]
            small_scale_preds = predictions_dict["small_scale_preds"]
            predictions = [large_scale_preds , medium_scale_preds , small_scale_preds]
            if ret:
                boxes_this_frame = post_processing.post_process([prediction[0:1] for prediction in predictions] ,
                                                            self.sorted_anchors)
                pp_utils.draw_predictions(cv2.resize(frame , (self.input_size,self.input_size)) , boxes_this_frame.numpy()[0] , self.class_names , waitkey_inp = False , invert_img = False)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print('FPS {:1f}'.format(1/(time.time() - stime)))
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def predict_video_input(self , model , video_file_path):
        if model is None:
            raise Exception("Detection model cannot be None")

        print(video_file_path)
        cap = cv2.VideoCapture(video_file_path)
        while(cap.isOpened()):
            stime = time.time()
            ret , frame = cap.read()
            preprocessed_frame = np.array(cv2.resize(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB) , (self.input_size,self.input_size)) , dtype = np.float32)
            inp_frame = preprocessed_frame.reshape(1,self.input_size,self.input_size,3) / 255.0
            predictions_dict = model(inp_frame , training = False)
            large_scale_preds = predictions_dict["large_scale_preds"]
            medium_scale_preds = predictions_dict["medium_scale_preds"]
            small_scale_preds = predictions_dict["small_scale_preds"]
            predictions = [large_scale_preds , medium_scale_preds , small_scale_preds]
            if ret:
                boxes_this_frame = post_processing.post_process([prediction[0:1] for prediction in predictions] ,
                                                            self.sorted_anchors)
                pp_utils.draw_predictions(cv2.resize(frame , (self.input_size,self.input_size)) , boxes_this_frame.numpy()[0] , self.class_names , waitkey_inp = False , invert_img = False)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print('FPS {:1f}'.format(1/(time.time() - stime)))
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def predict_batch_input(self , batch_generator , model ,show_out = False):
        pred_box_list = []
        gt_box_list = []
        all_images = []
        for index in range(self.num_batch):
            batch_images , org_images , batch_labels , org_labels , _ = batch_generator.load_data_for_test(index)
            predictions_dict = model(batch_images, training = False)
            large_scale_preds = predictions_dict["large_scale_preds"]
            medium_scale_preds = predictions_dict["medium_scale_preds"]
            small_scale_preds = predictions_dict["small_scale_preds"]
            predictions = [large_scale_preds , medium_scale_preds , small_scale_preds]
            for img_i in range(batch_images.shape[0]):
                boxes_this_image = post_processing.post_process([prediction[img_i:img_i+1] for prediction in predictions] ,
                                                                self.sorted_anchors)
                pred_box_list.append(boxes_this_image)
                gt_box_list.append(org_labels[img_i])
                all_images.append(org_images[img_i])
                if show_out:
                    pp_utils.draw_predictions(org_images[img_i] , boxes_this_image.numpy()[0] , self.class_names)

        return gt_box_list , pred_box_list , all_images
