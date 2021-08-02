import tensorflow as tf

from tf_CMT.Basic_blocks import CNN_Block, LocalPerceptionUnitLayer, InvertedResidualFFNLayer
from tf_CMT.LWMHSA import LightWeightMHSALayer

class CMT_Model(tf.keras.Model):
    def __init__(self, Block_num = [3,3,16,3], K=5, n_heads = 8, head_dim = 128, filters = 256, num_classes=10, usePosBias = True, output_logits = True):
        super(CMT_Model, self).__init__()
        self.stem            = CMT_Stem(filters = filters, strides = 2, kernel_size = 3 ,num_blocks = 2)
        self.DownSampleCNNs  = [tf.keras.layers.Conv2D(filters = filters, kernel_size = 2, strides = 2) for _ in range(len(Block_num))]
        self.CMT_Blocks_list = [[CMT_Block(n_heads     = n_heads,
                                           filters     = filters,
                                           kernel_size = 3,
                                           strides     = 2,
                                           K = K,
                                           usePosBias = usePosBias)
                                           for _ in range(Block_num[stage])]
                                           for stage in range(len(Block_num))] 
        self.global_pool     = tf.keras.layers.GlobalAveragePooling2D()
        self.FCLayer         = tf.keras.layers.Dense(num_classes)
        self.SoftmaxLayer    = tf.keras.layers.Softmax()
        self.output_logits   = output_logits
        
        
    def call(self, inputs):
        CMT_Input = self.stem(inputs)
        
        for DownSampleCNN, CMT_Blocks in zip(self.DownSampleCNNs,self.CMT_Blocks_list):
            CMT_Input = DownSampleCNN(CMT_Input)
            for Block in CMT_Blocks:
                CMT_Input = Block(CMT_Input)
        
        CMT_Out     = self.global_pool(CMT_Input) 
        FC_Out      = self.FCLayer(CMT_Out)
        if not self.output_logits:
            Output = self.SoftmaxLayer(FC_Out)
        else:
            Output = FC_Out
        return Output
        
class CMT_Stem(tf.keras.layers.Layer):
    def __init__(self, filters = 32, strides = 2, kernel_size = 3 ,num_blocks = 2,Name="CMT_Stem"):    
        self.DownSampleCNN = tf.keras.layers.Conv2D(filters=filters,kernel_size = kernel_size,strides = (strides,strides),padding='same')
        self.CNN_Blocks    = [CNN_Block(filters,kernel_size= kernel_size ,strides=1) for _ in range(num_blocks)]  
        super(CMT_Stem, self).__init__(name=Name)        

    def call(self, input):
        DownSampled_Output = self.DownSampleCNN(input)
        for CNN_Block in self.CNN_Blocks:
            DownSampled_Output = CNN_Block(DownSampled_Output)        
        return DownSampled_Output
    
class CMT_Block(tf.keras.layers.Layer):
    def __init__(self, n_heads = 8, filters = 32, kernel_size = 3,strides = 2, K = 5, usePosBias = False ,Name="CMT_Block"):
        self.LocalPerceptionUnit         = LocalPerceptionUnitLayer(kernel_size = 3, strides = 1)
        self.AttentionLayerNorm          = tf.keras.layers.LayerNormalization()
        self.LwMultiHeadSelfAttention    = LightWeightMHSALayer(n_heads = n_heads, head_dim = filters, dropout_rate=0.3, K = K,usePosBias=usePosBias)
        self.IRFFNLayerNorm              = tf.keras.layers.LayerNormalization()
        self.InvertedResidualFFN         = InvertedResidualFFNLayer(filters = filters)
        super(CMT_Block, self).__init__(name=Name)        

    def call(self, input):
        LPU_Out     = self.LocalPerceptionUnit(input)
        AttNormOut  = self.AttentionLayerNorm(LPU_Out)
        LwMHSAOut   = self.LwMultiHeadSelfAttention(AttNormOut)

        IRFFN_Input = LPU_Out + LwMHSAOut

        IRFFN_NormOut = self.IRFFNLayerNorm(IRFFN_Input)
        IRFFN_Out     = self.InvertedResidualFFN(IRFFN_NormOut)

        Output        = IRFFN_Input + IRFFN_Out
        return Output