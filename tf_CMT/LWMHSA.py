import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self,dropout_rate=0.3, usePosBias = False,**kwargs):
        self.dropout_rate = dropout_rate
        self.usePosBias    = usePosBias
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):  # Create the state of the layer (weights)
        if self.usePosBias:
            self.LearnableBias = self.add_weight(
                                                 shape       = (input_shape[0][1],input_shape[1][1]),
                                                 initializer = 'glorot_uniform',
                                                 trainable   = True,
                                                 name        = 'Learnable_PosBias'
                                                )      
        super(ScaledDotProductAttention, self).build(input_shape)

    def call(self, inputs):

        assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
        queries, keys, values = inputs

        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys)    != 'float32':  keys    = K.cast(keys, 'float32')
        if K.dtype(values)  != 'float32':  values  = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1])) # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        if self.usePosBias:
            scaled_matmul = scaled_matmul + self.LearnableBias

        softmax_out = K.softmax(scaled_matmul) # SoftMax
        
        # Dropout
        out     = K.dropout(softmax_out, self.dropout_rate)
        
        outputs = K.batch_dot(out, values)

        return outputs, softmax_out

    def compute_output_shape(self, input_shape):
        return input_shape
    
class LightWeightMHSALayer(tf.keras.layers.Layer):
    def __init__(self, n_heads, head_dim, dropout_rate=.3, K=5,usePosBias=False,**kwargs):
        self.num_heads    = n_heads
        assert head_dim % n_heads ==0, "Input feautures cannot be divided by multihead!!"
        self.head_dim     = head_dim // n_heads
        self.dropout_rate = dropout_rate
        self.DWCNN        = tf.keras.layers.DepthwiseConv2D(kernel_size=K, strides=(K,K))
        self.K            = K
        self.attention    = ScaledDotProductAttention(dropout_rate  = self.dropout_rate)
        ### Q, K, V weight matrix
        self.Query_Matrix  = tf.keras.layers.Dense(self.num_heads * self.head_dim)
        self.Key_Matrix    = tf.keras.layers.Dense(self.num_heads * self.head_dim)
        self.Value_Matrix  = tf.keras.layers.Dense(self.num_heads * self.head_dim)        
        
        super(LightWeightMHSALayer, self).__init__(**kwargs)

    def call(self, inputs):

        query          = self.Query_Matrix(inputs)
        ## Depth-wise CNN
        key            = self.DWCNN(inputs)
        value          = self.DWCNN(inputs)

        key            = self.Key_Matrix(key)
        value          = self.Value_Matrix(value)
        
        ## Flatten to linear
        linear_query   = tf.reshape(query,[-1,query.shape[1]*query.shape[2],self.num_heads * self.head_dim])
        linear_key     = tf.reshape(key,  [-1,key.shape[1]*key.shape[2],self.num_heads * self.head_dim])
        linear_value   = tf.reshape(value,[-1,value.shape[1]*value.shape[2],self.num_heads * self.head_dim])
        
        queries_multi_heads = tf.concat(tf.split(linear_query, self.num_heads, axis=2), axis=0)
        keys_multi_heads    = tf.concat(tf.split(linear_key,   self.num_heads, axis=2), axis=0)
        values_multi_heads  = tf.concat(tf.split(linear_value, self.num_heads, axis=2), axis=0)
        
        att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]
            
        att_out, att_score = self.attention(att_inputs)
                            
        outputs = tf.concat(tf.split(att_out, self.num_heads, axis=0), axis=2)
        
        ### Reshape back as image
        outputs = tf.reshape(outputs,[-1,query.shape[1],query.shape[2],self.num_heads * self.head_dim])
        
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape