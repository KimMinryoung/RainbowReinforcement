import math
import tensorflow as tf


class NoisyLinear():
    def __init__(self, conv_out, in_, out_, std_init=0.5, training=True):
        self.in_ = in_
        self.out_ = out_
        self.std_init = std_init

        self.input = conv_out 
        self.weight_mu = tf.Variable(tf.zeros([in_, out_],tf.float32), name="weight_mu")
        self.weight_sigma = tf.Variable(tf.zeros([in_, out_],tf.float32), name="weight_sigma")

        self.weight_epsilon = tf.constant(0.0,shape=[in_,out_],dtype=tf.float32)

        self.bias_mu = tf.Variable(tf.zeros([out_],tf.float32), name="bias_mu")
        self.bias_sigma = tf.Variable(tf.zeros([out_],tf.float32), name="bias_sigma")

        self.bias_epsilon = tf.constant(0.0, shape=[out_],dtype=tf.float32)

        self.result = tf.matmul(self.input, self.weight_mu) + self.bias_mu
        self.tr_result = tf.matmul(self.input,
                                      self.weight_mu + self.weight_sigma*self.weight_epsilon) + self.bias_mu + self.bias_sigma * self.bias_epsilon

        self.training = training

    def reset_variables(self):
        mu_range = 1 / math.sqrt(self.in_)

        self.weight_mu = tf.random_uniform([self.in_, self.out_], minval=-mu_range, maxval=mu_range, dtype=tf.float32)
        self.weight_sigma = tf.fill([self.in_, self.out_], self.std_init / math.sqrt(self.in_))

        self.bias_mu = tf.random_uniform([self.out_], minval=-mu_range, maxval=mu_range, dtype=tf.float32)
        self.weight_sigma = tf.fill([self.out_], self.std_init / math.sqrt(self.in_))

    def _scale_noise(self, size):
        self.x = tf.random.uniform([size])
        return tf.abs(self.x) * (tf.sqrt(tf.abs(self.x)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_)
        epsilon_out = self._scale_noise(self.out_)

        self.weight_epsilon = tf.constant(tf.tensordot(epsilon_out, epsilon_in, axes=0))
        self.bias_epsilon = tf.constant(epsilon_out)

class DQN():

    def __init__(self, args, action_space):
        self.action_space = action_space
        self.atoms = args.atoms

        self.inputs = tf.placeholder(tf.float32, [None,84,84,args.history_length], name="inputs")

        filter1 = tf.Variable(tf.random_normal([8, 8, args.history_length, 32], stddev=0.01))
        self.conv1 = tf.nn.conv2d(input=self.inputs, filter=filter1, strides=[1,4,4,1], padding='SAME')
        self.h1 = tf.nn.relu(self.conv1)
        filter2 = tf.Variable(tf.random_normal([4,4,32,64], stddev=0.01))
        self.conv2 = tf.nn.conv2d(input=self.h1, filter=filter2, strides=[1,2,2,1], padding='VALID')
        self.h2 = tf.nn.relu(self.conv2)
        filter3 = tf.Variable(tf.random_normal([3,3,64,64], stddev=0.01))
        self.conv3 = tf.nn.conv2d(input=self.h2, filter=filter3, strides=[1,1,1,1], padding='VALID')
        self.h3 = tf.nn.relu(self.conv3)

        self.fc_h_v = NoisyLinear(tf.reshape(self.h3,[-1,3136]), 3136, args.hidden_size, std_init=args.init_noisy_std)
        self.r_v = self.fc_h_v.tr_result
        self.fc_h_a = NoisyLinear(tf.reshape(self.h3,[-1,3136]), 3136, args.hidden_size, std_init=args.init_noisy_std)
        self.r_a = self.fc_h_a.tr_result
        self.h_fc_h_v = tf.nn.relu(self.r_v)
        self.h_fc_h_a = tf.nn.relu(self.r_a)
        
        self.fc_z_v = NoisyLinear(self.h_fc_h_v, args.hidden_size, self.atoms, std_init=args.init_noisy_std)
        self.z_v = self.fc_z_v.tr_result
        self.fc_z_a = NoisyLinear(self.h_fc_h_a, args.hidden_size, action_space * self.atoms, std_init=args.init_noisy_std)
        self.z_a = self.fc_z_a.tr_result
        self.h_fc_z_v = tf.nn.relu(self.z_v)
        self.h_fc_z_a = tf.nn.relu(self.z_a)
        
        self.v = tf.reshape(self.h_fc_z_v,[-1,1,self.atoms])
        self.a = tf.reshape(self.h_fc_z_a,[-1,self.action_space,self.atoms])
        
        self.q = self.v + self.a - tf.reduce_mean(self.a,axis=1,keep_dims=True)
        self.action = tf.nn.softmax(self.q)
        self.action_log = tf.nn.log_softmax(self.q)
        
    def train(self):
        self.fc_h_v.training = True
        self.fc_h_a.training = True
        self.fc_z_v.training = True
        self.fc_z_a.training = True
        
    def eval(self):
        self.fc_h_v.training = False
        self.fc_h_a.training = False
        self.fc_z_v.training = False
        self.fc_z_a.training = False
        
    def reset_noise(self):
        self.fc_h_v.reset_noise
        self.fc_h_a.reset_noise
        self.fc_z_v.reset_noise
        self.fc_z_a.reset_noise