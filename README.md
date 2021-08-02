# tf-CMT
Personal implementation of CMT: Convolutional Neural Networks Meet Vision Transformers in tensorflow.  
Paper: https://arxiv.org/abs/2107.06263  
All suggestions are welcome.

## Usage

An example of model is shown below:
```python
import tensorflow
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
                 
test_image = tf.random.normal([1, 224, 224, 3])
model(test_image) # Output shape is (1,10)
```
Beware:  
For each stage, the input will be downsampled by a 2x2 2D-CNN Layer with stride=2.  
Please be aware of your input sizes at each stage.  

## Evalutation
The CMT model is evalutated by MNIST handwritten dataset, and reached val_acc of 0.9892 at epoch=5.  
Details are shown in CMT_Demo.ipynb.  

