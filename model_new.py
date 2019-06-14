import math
import tensorflow as tf
import numpy as np

class DQN:
  def __init__(self, args, action_size):
    self.atoms = args.atoms
    self.action_size = action_size
    self.lr = args.lr
    
    self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
    self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
          
    self.target_Q = tf.placeholder(tf.float32, [None], name="target")