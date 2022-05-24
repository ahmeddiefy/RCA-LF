import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

kernel_size = 3
f = 512
f_input=64
n_res_blocks=10
n_res_groups=1
weights = []
w_init = tf.variance_scaling_initializer(scale=1., mode='fan_avg', distribution="uniform")
b_init = tf.zeros_initializer()




def conv2d(x, f_in, f_out, k, name):

    conv_w = tf.get_variable(name + "_w" , [k,k,f_in,f_out], initializer=w_init)
    conv_b = tf.get_variable(name + "_b" , [f_out], initializer=b_init)
    weights.append(conv_w)
    weights.append(conv_b)
    return tf.nn.bias_add(tf.nn.conv2d(x, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)                           


def channel_attention(x, name):
    y = tf.compat.v1.keras.layers.GlobalAvgPool2D()(x)
    y = tf.expand_dims(y, axis=1)
    y = tf.expand_dims(y, axis=1)
    conv_name = "conv2d-1" + "_" + name
    y1 = conv2d(y, f_in=f, f_out=f_input, k=1, name=conv_name)
    y2 = tf.nn.relu(y1)
    conv_name = "conv2d-2" + "_" + name
    y3 = conv2d(y2, f_in=f_input, f_out=f, k=1, name=conv_name)
    y4 = tf.nn.sigmoid(y3)

    return tf.multiply(y4, x)
    
            
            
def residual_channel_attention_block(x, name):
    skip_conn = x  
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-2" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.add(x , skip_conn)

    return channel_attention(x, name+"_CA_")
    
    
    
    
def residual_group(x, name):
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="residual_group-head" + name)
    skip_conn = x

    for i in range(n_res_blocks):
        x = residual_channel_attention_block(x, name=name + "_" + str(i))
        
    conv_name = "rg-conv-" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    return tf.add(x , skip_conn) 
            
            
            
            
            
            
def residual_channel_attention_network(x):
    # 1. head
    x = conv2d(x, f_in=f_input, f_out=f, k=kernel_size, name="conv2d-head_0")
    head = x

    # 2. body
    x = head
    for i in range(n_res_groups):
        x = residual_group(x, name=str(i) )

    body = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="conv2d-body")
    body = tf.add(body , head)

    tail = conv2d(body, f_in=f, f_out=f_input, k=kernel_size, name="conv2d-tail")  

    return tail
         
def model(input_tensor):
    with tf.device("/gpu:0"):
        tensor = None
        tensor = residual_channel_attention_network(input_tensor)
        print(tensor.shape)

		
        return tensor, weights

