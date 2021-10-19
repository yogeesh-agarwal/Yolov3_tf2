import pickle
import numpy as np
import tensorflow as tf
from darknet53 import Darknet53
from layers import InterConvBlocks
from layers import ConvBlock

class Yolov3(tf.keras.Model):
    def __init__(self , base_grid_size , num_classes , grid_scales , weights_path , bn_weights_path):
        super(Yolov3, self).__init__(name = "Yolov3")
        self.base_grid_height = base_grid_size
        self.bn_weights_path = bn_weights_path
        self.base_grid_width = base_grid_size
        self.weights_path = weights_path
        self.grid_scales = grid_scales
        self.num_classes = num_classes
        self.num_anchors_per_scale = 3
        self.conv_index = 52
        self.icb_index = 0

        self.darknet_classifier = Darknet53(self.weights_path , self.bn_weights_path)
        self.icb1 = InterConvBlocks(self.conv_index,
                                    [[1,512] , [3,1024] , [1,512] , [3,1024] , [1,512] , [3,1024]],
                                    self.icb_index)
        self.conv_index += 6
        self.icb_index += 1
        self.large_scale_preds = ConvBlock(index = self.conv_index ,
                                           kernel_size = 1 ,
                                           num_filters = 3*(5 + self.num_classes) ,
                                           do_bias = True ,
                                           do_lr = False)
        self.conv_index += 1
        self.conv_85 = ConvBlock(index = self.conv_index ,
                                 kernel_size = 1,
                                 num_filters = 256)
        self.conv_index += 1
        self.icb2 = InterConvBlocks(self.conv_index,
                                    [[1,256] , [3,512] , [1,256] , [3,512] , [1,256] , [3,512]],
                                    self.icb_index)
        self.conv_index += 6
        self.icb_index += 1
        self.medium_scale_preds = ConvBlock(index = self.conv_index ,
                                            kernel_size = 1,
                                            num_filters = 3*(5 + self.num_classes),
                                            do_bias = True,
                                            do_lr = False)
        self.conv_index += 1
        self.conv_97 = ConvBlock(index = self.conv_index ,
                                kernel_size = 1,
                                num_filters = 128)
        self.conv_index += 1
        self.icb3 = InterConvBlocks(self.conv_index,
                                    [[1,128] , [3,256] , [1,128] , [3,256] , [1,128] , [3,256]],
                                    self.icb_index)
        self.conv_index += 6
        self.icb_index += 1
        self.small_scale_preds = ConvBlock(index = self.conv_index ,
                                            kernel_size = 1,
                                            num_filters = 3*(5 + self.num_classes),
                                            do_bias = True,
                                            do_lr = False)

        # initialize own metrics to keep track of loss and map.
        self.loss_tracker = tf.keras.metrics.Mean(name = "training_loss")
        

    def upsampling_block(self , inputs , output_shape):
        new_height = output_shape[1]
        new_width = output_shape[2]
        return tf.image.resize(inputs , (new_height , new_width))

    def call(self , inputs):
        concat_1 , concat_2 , conv_75 = self.darknet_classifier(inputs)
        conv_80 , conv_81 = self.icb1(conv_75)
        large_scale_preds = tf.reshape(self.large_scale_preds(conv_81) , [-1 , self.base_grid_height * self.grid_scales[0] , self.base_grid_width * self.grid_scales[0] , self.num_anchors_per_scale , (4 + 1 + self.num_classes)])

        conv_85 = self.conv_85(conv_80)
        upsampling_shape_1 = concat_2.get_shape().as_list()
        upsample_1 = self.upsampling_block(conv_85 , upsampling_shape_1)
        route_1 = tf.concat([upsample_1 , concat_2] , axis = -1)
        conv_92 , conv_93 = self.icb2(route_1)
        medium_scale_preds = tf.reshape(self.medium_scale_preds(conv_93) , [-1 , self.base_grid_height * self.grid_scales[1] , self.base_grid_width * self.grid_scales[1] , self.num_anchors_per_scale , (4 + 1 + self.num_classes)])

        conv_97 = self.conv_97(conv_92)
        upsampling_shape_2 = concat_1.get_shape().as_list()
        upsample_2 = self.upsampling_block(conv_97 , upsampling_shape_2)
        route_2 = tf.concat([upsample_2 , concat_1] , axis = -1)
        conv_104 , conv_105 = self.icb3(route_2)
        small_scale_preds = tf.reshape(self.small_scale_preds(conv_105) , [-1 , self.base_grid_height * self.grid_scales[1] , self.base_grid_width * self.grid_scales[1] , self.num_anchors_per_scale , (4 + 1 + self.num_classes)])

        all_predictions = [large_scale_preds , medium_scale_preds , small_scale_preds]
        return all_predictions

    def build_graph(self):
        x = tf.keras.Input(shape = (416,416,3))
        return tf.keras.Model(inputs = [x] , outputs = self.call(x))


def test_model():
    yolo = Yolov3(13 , 2 , [1,2,4] ,
                  "/home/yogeesh/yogeesh/object_detection/yolov3/data/conv_weights.pickle",
                  "/home/yogeesh/yogeesh/object_detection/yolov3/data/bn_weights.pickle")
    yolo.build(input_shape = [1,416,416,3])
    yolo.build_graph().summary()
    tf.keras.utils.plot_model(yolo.build_graph(), to_file='./arch_png/yolov3.png', show_shapes=True, show_dtype=False,
                              show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96,
                              layer_range=None)

test_model()
