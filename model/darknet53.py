import pickle
import numpy as np
import tensorflow as tf
from Yolov3_tf2.model.layers import ConvBlock , ResidualBlock , InterConvBlocks

class Darknet53(tf.keras.Model):
    def __init__(self, darknet53_weights_file , darknet53_bn_file):
        super(Darknet53, self).__init__(name = "DarkNet53")
        self.darknet53_weights = self.load_pickle(darknet53_weights_file)
        self.darknet53_bn = self.load_pickle(darknet53_bn_file)
        self.conv_index = 0
        self.rb_index = 0
        self.bn_index = 0

        self.conv1 = ConvBlock(index = self.conv_index , conv_weights = self.get_weights() , bn_weights = self.get_bn_weights() ,
                               darknet = True)
        self.conv2 = ConvBlock(index = self.conv_index , conv_weights = self.get_weights() , bn_weights = self.get_bn_weights() ,
                               downsample = True , kernel_size = 3 , darknet = True)
        self.rb1 = ResidualBlock(self.conv_index ,
                          [self.get_weights() , self.get_weights()] ,
                          [self.get_bn_weights() , self.get_bn_weights()] ,
                           self.rb_index)
        self.rb_index += 1
        self.conv3 = ConvBlock(index = self.conv_index , conv_weights = self.get_weights() , bn_weights = self.get_bn_weights() ,
                               downsample = True , kernel_size =3 , darknet = True)
        self.rb2 = []
        for i in range(2):
            self.rb2.append(ResidualBlock(self.conv_index ,
                              [self.get_weights() , self.get_weights()] ,
                              [self.get_bn_weights() , self.get_bn_weights()] ,
                               self.rb_index))
            self.rb_index += 1

        self.conv4 = ConvBlock(index = self.conv_index , conv_weights = self.get_weights() , bn_weights = self.get_bn_weights(),
                               downsample = True , kernel_size = 3 , darknet = True)
        self.rb3 = []
        for i in range(8):
            self.rb3.append(ResidualBlock(self.conv_index ,
                              [self.get_weights() , self.get_weights()] ,
                              [self.get_bn_weights() , self.get_bn_weights()] ,
                               self.rb_index))
            self.rb_index += 1

        self.conv5 = ConvBlock(index = self.conv_index , conv_weights = self.get_weights() , bn_weights = self.get_bn_weights(),
                               downsample = True , kernel_size = 3 , darknet = True)
        self.rb4 = []
        for i in range(8):
            self.rb4.append(ResidualBlock(self.conv_index ,
                              [self.get_weights() , self.get_weights()] ,
                              [self.get_bn_weights() , self.get_bn_weights()] ,
                               self.rb_index))
            self.rb_index += 1

        self.conv6 = ConvBlock(index = self.conv_index , conv_weights = self.get_weights() , bn_weights = self.get_bn_weights(),
                               downsample = True , kernel_size = 3 , darknet = True)
        self.rb5 = []
        for i in range(4):
            self.rb5.append(ResidualBlock(self.conv_index ,
                              [self.get_weights() , self.get_weights()] ,
                              [self.get_bn_weights() , self.get_bn_weights()] ,
                               self.rb_index))
            self.rb_index += 1

    def load_pickle(self, file):
        with open(file , "rb") as content:
            return pickle.load(content)

    def get_weights(self):
        weights = self.darknet53_weights["conv_{}".format(self.conv_index)]
        self.conv_index += 1
        return weights

    def get_bn_weights(self):
        bn_weights = self.darknet53_bn["conv_{}".format(self.bn_index)]
        self.bn_index += 1
        return bn_weights

    def call(self , inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.rb1(x)
        x = self.conv3(x)
        for i in range(2):
            x = self.rb2[i](x)

        x = self.conv4(x)
        for i in range(8):
            x = self.rb3[i](x)

        shortcut_1 = x
        x = self.conv5(x)
        for i in range(8):
            x = self.rb4[i](x)

        shortcut_2 = x
        x = self.conv6(x)
        for i in range(4):
            x = self.rb5[i](x)

        return shortcut_1 , shortcut_2 , x

    def build_graph(self):
        x = tf.keras.Input(shape = (256,256,3))
        return tf.keras.Model(inputs = [x] , outputs = self.call(x))

def test_model():
    darknet = Darknet53("/home/yogeesh/yogeesh/object_detection/yolov3/data/conv_weights.pickle",
                        "/home/yogeesh/yogeesh/object_detection/yolov3/data/bn_weights.pickle")
    darknet.build(input_shape = [1,256,256,3])
    darknet.build_graph().summary()
    tf.keras.utils.plot_model(darknet.build_graph(), to_file='./arch_png/darknetv3.png', show_shapes=True, show_dtype=False,
                              show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96,
                              layer_range=None)
