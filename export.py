import tensorflow as tf

from tf_CMT.model import CMT_Model

model = CMT_Model(Block_num     = [3,12], # Number of CMT_Blocks in each stage
                  K             = 2,      # HyperParam to reduce the complexity of self-attention to O(N^2/k^2)
                  n_heads       = 4,      # Number of heads
                  head_dim      = 256,    # The latent dimension of self-attention
                  filters       = 256,    # Number of filters of CNNs
                  num_classes   = 10,     # Number of output classes
                  usePosBias    = True,   # Use learnable positional bias 
                  output_logits = True    # Output logits or not
                 )
                 
# test_image = tf.random.normal([1, 224, 224, 3])

try:
    # for keras
    tf.saved_model.save(model, "./export_model")

    # for tf.Model
    # tf.saved_model.save(
    #     model,
    #     export_dir: "./export_model",
    #     signatures=None,
    # )
    print("[JIT] Export model by TensorFlow successed.")
except:
    print("[JIT] Export model by TensorFlow failed.")
