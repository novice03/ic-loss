import torch
import tensorflow as tf
import numpy as np

def icl(mu, labels, icl_a = 5.0, icl_b = 0.1):
    diff = mu.unsqueeze(0) - mu.unsqueeze(1)
    diff = diff.norm(dim = -1)
    sq_loss = diff.pow(2)
    lap_kernel = torch.exp(icl_a - icl_b * diff)
    
    s = torch.zeros_like(labels).float()
    c = 3
    for i in range(c):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue    
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)
    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    loss = (sq_loss * s_prod)[c_diff == True].sum()
    loss += (c-1) * (lap_kernel * s_prod)[c_diff == False].sum()
    
    return loss

def icl_tf(mu, labels, icl_a = 5.0, icl_b = 0.1):
    diff = tf.expand_dims(mu, 0) - tf.expand_dims(mu, 1)
    diff = tf.norm(diff, axis = -1)
    sq_loss = tf.pow(diff, 2)
    lap_kernel = tf.exp(icl_a - icl_b * diff)
    
    s = tf.cast(tf.zeros_like(labels), dtype = tf.float32)
    c = 3
    for i in range(c):
        index = labels == i
        n = np.asscalar(tf.reduce_sum(tf.cast(index, dtype = tf.int16)).numpy())
        if n < 1e-9:
            continue  
        s = tf.where(index, 1.0/n, s)
    s_prod = tf.expand_dims(s, 1) * tf.expand_dims(s, 0)
    c_diff = tf.expand_dims(labels, 1) != tf.expand_dims(labels, 0)
    loss = tf.reduce_sum((sq_loss * s_prod)[c_diff == True])
    loss += tf.reduce_sum((c-1) * (lap_kernel * s_prod)[c_diff == False])
    
    return loss   