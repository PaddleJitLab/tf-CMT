import tensorflow as tf


class CNN_Block(tf.keras.layers.Layer):
    def __init__(self, filters = 32, kernel_size = 3,strides = 2,Name="CNN_Block"):
        self.CNN_Layer = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides,padding='same')
        self.ActLayer  = tf.keras.activations.gelu
        self.BNLayer   = tf.keras.layers.BatchNormalization()
        super(CNN_Block, self).__init__(name=Name)        

    def call(self, input):
        CNN_Output = self.CNN_Layer(input)
        Act_Output = self.ActLayer(CNN_Output)
        BN_Output  = self.BNLayer(Act_Output)
        return BN_Output
    
class LocalPerceptionUnitLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size = 3,strides = 1 ,Name="LocalPerceptionUnit"):
        self.DWCNN = tf.keras.layers.DepthwiseConv2D(kernel_size = kernel_size, strides=(strides, strides), padding = 'same')
        super(LocalPerceptionUnitLayer, self).__init__(name=Name)        
        
    def call(self, input):
        DWCNN_Out     = self.DWCNN(input)
        Output        = input + DWCNN_Out
        return Output
    
class InvertedResidualFFNLayer(tf.keras.layers.Layer):
    def __init__(self, filters = 256):
        
        self.FirstCNNBlock   = CNN_Block(filters = filters, kernel_size = 1,strides = 1 ,Name="FirstCNN_Block")
        self.SecondCNNBlock  = CNN_Block(filters = filters, kernel_size = 1,strides = 1 ,Name="SecondCNN_Block")
        self.DWCNNLayer      = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(1,1),padding='same')
        self.LastBN          = tf.keras.layers.BatchNormalization()

        super(InvertedResidualFFNLayer, self).__init__()

    def call(self, inputs):

        FirstCNN_Out    = self.FirstCNNBlock(inputs)
        DW_Out          = self.DWCNNLayer(FirstCNN_Out)
        SecondCNN_Input = FirstCNN_Out + DW_Out
        SecondCNN_Out   = self.SecondCNNBlock(DW_Out+FirstCNN_Out)
        outputs         = self.LastBN(SecondCNN_Out)
        
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape