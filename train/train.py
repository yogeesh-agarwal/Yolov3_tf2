import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from Yolov3_tf2.model.yolov3 import Yolov3
from Yolov3_tf2.loss.loss import Yolov3Loss
from Yolov3_tf2 import utils as comman_utils
import tensorflow.keras.callbacks as callbacks_module
from Yolov3_tf2.preprocessing.preprocessing import DataGenerator
from Yolov3_tf2.postprocessing import tf_postprocessing as post_processing

class Train():
    def __init__(self):
        self.verbose = 1
        self.is_norm = True
        self.batch_size = 4
        self.init_lr = 0.0001
        self.input_size = 416
        self.num_epochs = 1000
        self.is_augment = True
        self.base_grid_size = 13
        self.grid_scales = [1,2,4]
        self.load_pretrain = False
        self.val_inst_count = 10
        self.custom_training = True
        self.logs_dir = "../logs/"
        self.save_dir = "../saved_models/"
        self.labels = ["Face" , "Non_Face"]
        self.train_file = "../data/wider_training_data.pickle"
        self.val_file = "../data/wider_validation_data.pickle"
        self.darknet53_bn = None  # or "../data/bn_weights.pickle" (for pretrained darknet classifier)
        self.train_image_names = "../data/wider_images_names.pickle"
        self.val_image_names = "../data/wider_val_images_names.pickle"
        self.darknet53_weights = None # or "../data/conv_weights.pickle" (for pretrained darknet classifier)
        self.val_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_val/WIDER_val/images/"
        self.train_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images/"

        self.anchors = np.array([[0.17028831 , 0.35888521],
                            [0.05563053 , 0.09101727],
                            [0.11255733 , 0.21961425],
                            [0.0347448  , 0.06395953],
                            [0.32428802 , 0.42267646],
                            [0.47664651 , 0.65827237],
                            [0.21481797 , 0.20969635],
                            [0.07297461 , 0.14739788],
                            [0.11702667 , 0.11145465]] , dtype = np.float32).reshape(9,2) * self.input_size

        self.sorted_anchors = comman_utils.sort_anchors(self.anchors)

        self.train_summary_writer = tf.summary.create_file_writer(self.logs_dir)
        tf.summary.experimental.set_step(0)

        # define training data generator
        self.train_data_generator = DataGenerator(self.input_size ,
                                             self.base_grid_size ,
                                             self.grid_scales ,
                                             self.sorted_anchors ,
                                             self.train_data_path ,
                                             self.train_file ,
                                             self.train_image_names ,
                                             self.labels ,
                                             self.is_norm ,
                                             self.is_augment,
                                             self.batch_size,
                                             100)
        self.num_batches = len(self.train_data_generator)

        #define validation data generator
        self.val_data_generator = DataGenerator(self.input_size ,
                                             self.base_grid_size ,
                                             self.grid_scales ,
                                             self.sorted_anchors ,
                                             self.val_data_path ,
                                             self.val_file ,
                                             self.val_image_names ,
                                             self.labels ,
                                             self.is_norm ,
                                             False,
                                             self.batch_size,
                                             self.val_inst_count)

        #define yolov3 model ,
        # make sure to provide weight and bn_weights path if load_pretrain is True else darknet53 will be initialzed from scratch.
        self.face_detector = Yolov3(self.sorted_anchors ,
                               len(self.labels) ,
                               self.grid_scales ,
                               self.base_grid_size ,
                               load_pretrain = self.load_pretrain ,
                               weights_path =  self.darknet53_weights ,
                               bn_weights_path = self.darknet53_bn)

    def define_losses(self):
        # define 3 losses.
        # large_obj_loss : [13,13]
        # medium_obj_loss : [26,26]
        # small_obj_loss : [52,52]
        self.large_obj_loss = Yolov3Loss(self.base_grid_size , self.grid_scales[0] , len(self.labels) , self.sorted_anchors[:3], summary_writer = self.train_summary_writer , name = "large_obj_loss")
        self.medium_obj_loss = Yolov3Loss(self.base_grid_size , self.grid_scales[1] , len(self.labels) , self.sorted_anchors[3:6], summary_writer = self.train_summary_writer , name = "medium_obj_loss")
        self.small_obj_loss = Yolov3Loss(self.base_grid_size , self.grid_scales[2] , len(self.labels) , self.sorted_anchors[6:], summary_writer = self.train_summary_writer , name = "small_obj_loss")

    def gen_callbacks(self , tb_ld , cp_path):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = tb_ld,
                                                              histogram_freq = 1,
                                                              update_freq = "batch")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = cp_path,
                                                                 verbose = 1,
                                                                 mode = "min",
                                                                 save_best_only = True,
                                                                 monitor = "train_loss",
                                                                 save_weights_only = True)

        reduce_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = "train_loss",
                                                                          factor = 0.5,
                                                                          patience = 20,
                                                                          verbose = 1,
                                                                          mode = "min",
                                                                          min_lr = 1e-6)
        return [tensorboard_callback , checkpoint_callback , reduce_on_plateau_callback]

    def load_pretrained_detector(self):
        if not os.listdir(self.save_dir):
            print("No trained model found, training from scratch")
        else:
            latest_chkpnt = tf.train.latest_checkpoint(self.save_dir)
            self.face_detector.load_weights(latest_chkpnt)
            print("Partial trained model found , loading weights.")

    def configure_callbacks(self , callbacks):
        if not isinstance(callbacks , callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(callbacks ,
                                                      add_history = True,
                                                      add_progbar = False ,
                                                      model = self.face_detector ,
                                                      verbose = self.verbose ,
                                                      epochs = self.num_epochs ,
                                                      steps = len(self.train_data_generator))
        return callbacks

    @tf.function
    def forward_step(self , data , training = True):
        images , ground_truths = data
        large_obj_gt , medium_obj_gt , small_obj_gt = ground_truths
        all_predictions = self.face_detector(images , training = training)
        loss = self.face_detector.compiled_loss([large_obj_gt , medium_obj_gt , small_obj_gt],
                                  all_predictions ,
                                  regularization_losses = self.face_detector.losses)

        return all_predictions , loss

    def process_predictions(self , all_predictions):
        boxes_this_data = []
        small_scale_preds = all_predictions["small_scale_preds"]
        large_scale_preds = all_predictions["large_scale_preds"]
        medium_scale_preds = all_predictions["medium_scale_preds"]
        num_instances = tf.shape(large_scale_preds)[0]
        predictions = [large_scale_preds , medium_scale_preds , small_scale_preds]
        for index in tf.range(num_instances):
            boxes = post_processing.post_process([prediction[index:index+1] for prediction in predictions] ,
                                                  self.sorted_anchors ,
                                                  num_classes = len(self.labels))
            boxes_this_data.append(boxes)
        return boxes_this_data

    @tf.function
    def train_step(self  , data):
        # override this fucntion to implement custom logic for one training step (i.e. per batch)
        images , ground_truths = data
        large_obj_gt , medium_obj_gt , small_obj_gt = ground_truths

        with tf.GradientTape() as tape:
            # run forward pass under gradient_tape to log the operations implemented , will be used for back propagation.
            all_predictions , loss = self.forward_step(data)

        # apply gradients on trainable vars.
        trainable_variables = self.face_detector.trainable_variables
        gradients = tape.gradient(loss , trainable_variables)
        self.face_detector.optimizer.apply_gradients(zip(gradients , trainable_variables))

        # compute metrics
        self.face_detector.train_loss_tracker.update_state(loss)
        return {"train_loss" : self.face_detector.train_loss_tracker.result()}

    def test_step(self , data):
        # override this function to implement custom logic for one test step (i.e. per batch)
        images , ground_truth_data = data
        ground_truths , org_labels = ground_truth_data
        all_predictions , loss = self.forward_step([images , ground_truths] , training = False)
        boxes_this_batch = self.process_predictions(all_predictions)
        self.face_detector.val_mAP.update_state(org_labels , boxes_this_batch)
        self.face_detector.val_loss_tracker.update_state(loss)
        return {"loss" : self.face_detector.val_loss_tracker.result() , "mAP" : self.face_detector.val_mAP.result()}

    def custom_training_loop(self , callbacks):
        callbacks.on_train_begin()
        metric_names = ["train_loss" , "val_loss" , "val_mAP"]
        for epoch in range(self.num_epochs):
            train_log = None
            self.face_detector.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            print("Epoch {}/{}".format(epoch , self.num_epochs))
            prog_bar = tf.keras.utils.Progbar(self.num_batches , stateful_metrics = metric_names)
            step = 0
            while step < self.num_batches:
                callbacks.on_train_batch_begin(step)
                image_data_this_batch , label_data_this_batch , org_labels_this_batch = self.train_data_generator.__getitem_custom__(step)
                train_log = self.train_step([image_data_this_batch , label_data_this_batch])
                if step < self.num_batches-1:
                    prog_bar.update(step+1 , values = [("train_loss" , train_log["train_loss"])])
                else:
                    updated_values = [("train_loss" , train_log["train_loss"])]
                callbacks.on_train_batch_end(step+1 , train_log)
                step += 1

            for val_batch_idx in range(len(self.val_data_generator)):
                val_image_data_this_batch , val_label_data_this_batch , org_val_labels_this_batch = self.val_data_generator.__getitem_custom__(val_batch_idx)
                val_log = self.test_step([val_image_data_this_batch , [val_label_data_this_batch , org_val_labels_this_batch]])

            updated_values += [("val_loss" , val_log["loss"]) ,
                               ("val_mAP" , val_log["mAP"])]
            prog_bar.update(step , values = updated_values)
            callbacks.on_epoch_end(epoch , train_log)


    def train(self):
        callbacks = self.gen_callbacks(self.logs_dir+"/loss" , self.save_dir+"yolov3.ckpt")
        self.load_pretrained_detector()
        self.define_losses()
        # compile the model
        self.face_detector.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.init_lr) ,
                              loss = {"large_scale_preds" : self.large_obj_loss ,
                                      "medium_scale_preds" : self.medium_obj_loss ,
                                      "small_scale_preds" : self.small_obj_loss})

        # invoke custom training loop if custom_training flag is True:
        # custom training is configured to give model.fit kind off output display but additionally it also gives flexibility of running validation (mAP) .
        # which is not possible for now in graph mode.
        # so training happens in Graph mode , validation happens in Eager mode.
        if self.custom_training:
            #configure callback container.
            callbacks = self.configure_callbacks(callbacks)
            # call the training loop
            self.custom_training_loop(callbacks)

        else:
            # call model.fit
            # validation is not supported in fit method.
            self.face_detector.fit(self.train_data_generator ,
                              epochs = self.num_epochs,
                              use_multiprocessing = True,
                              initial_epoch = 0,
                              callbacks = callbacks)
        print("Training Completed")

def train_model():
    train_obj = Train()
    train_obj.train()

if __name__ == '__main__':
    train_model()
