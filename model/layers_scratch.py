import pickle
import numpy as np
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self ,
                 alpha = 0.1 ,
                 index = None ,
                 do_lr = True ,
                 do_bias = False ,
                 darknet = False ,
                 bn_weights = None ,
                 name = "Conv_blk" ,
                 downsample = False ,
                 kernel_size = None ,
                 num_filters = None ,
                 conv_weights = None ,
                 **kwargs):

        self.index = index
        self.layer_name = name + "_" + str(index)
        if darknet:
            self.layer_name += "_darknet"
        super(ConvBlock, self).__init__(name = self.layer_name , **kwargs)
        self.alpha = alpha
        self.do_lr = do_lr
        self.do_bias = do_bias
        self.bn_weights = bn_weights
        self.downsample = downsample
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_weights = conv_weights

    def fixed_padding(self , inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],[pad_beg, pad_end], [0, 0]])
        return padded_inputs

    def build(self, input_shape):
        self.padding = "SAME"
        self.strides = [1,1,1,1]
        if self.downsample:
            self.strides = [1,2,2,1]
            self.padding = "VALID"

        if not self.kernel_size:
            raise Exception("kernel Size is not available")
        # print((self.kernel_size, self.kernel_size , input_shape[-1] , self.num_filters))
        self.kernel = self.add_weight(shape = [self.kernel_size, self.kernel_size , input_shape[-1] , self.num_filters],
                                      initializer  = "random_normal" ,
                                      dtype = tf.float32 ,
                                      trainable = True,
                                      name = "weights_{}".format(self.index))
        if self.do_bias:
            self.bias = self.add_weight(shape = ([self.num_filters]),
                                        initializer = "random_normal" ,
                                        dtype = tf.float32 ,
                                        trainable = True,
                                        name = "biases_{}".format(self.index))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        super(ConvBlock , self).build(input_shape = input_shape)

    def call(self , inputs):
        if self.downsample:
            inputs = self.fixed_padding(inputs , self.kernel_size)
        x = tf.nn.conv2d(inputs, self.kernel , self.strides , self.padding)
        if not self.do_bias:
            x = self.batch_norm(x , training = True)
        else:
            x = tf.nn.bias_add(x , self.bias)

        if self.do_lr:
            x = tf.nn.leaky_relu(x , alpha = self.alpha)

        return x

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self , index , rb_index , kernel_sizes , num_filters , name = "residual_blk" , **kwargs):
        layer_name = name + str(rb_index)
        super(ResidualBlock , self).__init__(name = layer_name , **kwargs)
        self.conv1 = ConvBlock(index = index , kernel_size = kernel_sizes[0] , num_filters = num_filters[0] , darknet = True)
        self.conv2 = ConvBlock(index = index+1 ,  kernel_size = kernel_sizes[1] , num_filters = num_filters[1]  , darknet = True)

    def call(self, inputs):
        left_br = inputs
        right_br = self.conv1(inputs)
        right_br = self.conv2(right_br)
        return tf.add(left_br , right_br)

class InterConvBlocks(tf.keras.layers.Layer):
    def  __init__(self , index , weight_shapes , icb_index , name = "intermediate_convs_block" , **kwargs):
        layer_name = name + str(icb_index)
        super(InterConvBlocks , self).__init__(name = layer_name , **kwargs)
        if len(weight_shapes) != 6:
            raise Exception("insufficient data shapes")

        self.conv1 = ConvBlock(index = index , kernel_size = weight_shapes[0][0] , num_filters = weight_shapes[0][1])
        self.conv2 = ConvBlock(index = index+1 , kernel_size = weight_shapes[1][0] , num_filters = weight_shapes[1][1])
        self.conv3 = ConvBlock(index = index+2 , kernel_size = weight_shapes[2][0] , num_filters = weight_shapes[2][1])
        self.conv4 = ConvBlock(index = index+3 , kernel_size = weight_shapes[3][0] , num_filters = weight_shapes[3][1])
        self.conv5 = ConvBlock(index = index+4 , kernel_size = weight_shapes[4][0] , num_filters = weight_shapes[4][1])
        self.conv6 = ConvBlock(index = index+5 , kernel_size = weight_shapes[5][0] , num_filters = weight_shapes[5][1])

    def call(self , inputs , training = True):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        feed_forward_shortcut = x
        x = self.conv6(x)
        return feed_forward_shortcut , x
