import math
import tensorflow as tf


class NoisyLinear():
    def __init__(self, in_, out_, std_init=0.5, training=True):
        self.in_ = in_
        self.out_ = out_
        self.std_init = std_init

        self.input = tf.placeholder(tf.float32, [None, self.in_], name="input")

        self.weight_mu = tf.Variable(tf.zeros([in_, out_],tf.float32), name="weight_mu")
        self.weight_sigma = tf.Variable(tf.zeros([in_, out_],tf.float32), name="weight_sigma")

        self.weight_epsilon = tf.constant(0.0,shape=[in_,out_],dtype=tf.float32)

        self.bias_mu = tf.Variable(tf.zeros([out_],tf.float32), name="bias_mu")
        self.bias_sigma = tf.Variable(tf.zeros([out_],tf.float32), name="bias_sigma")

        self.bias_epsilon = tf.constant(0.0, shape=[out_],dtype=tf.float32)

        self.result = tf.matmul(self.input, self.weight_mu) + self.bias_mu
        self.tr_result = tf.matmul(self.input,
                                      self.weight_mu + self.weight_sigma*self.weight_epsilon) + self.bias_mu + self.bias_sigma * self.bias_epsilon

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        self.training = training

    def reset_variables(self):
        mu_range = 1 / math.sqrt(self.in_)

        self.weight_mu = tf.random_uniform([self.out_, self.in_], minval=-mu_range, maxval=mu_range, dtype=tf.float32)
        self.weight_sigma = tf.fill([self.out_, self.in_], self.std_init / math.sqrt(self.in_))

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

    def forward(self, input):
        
        if self.training:
            self.sess.run(self.tr_result, feed_dict={self.input: input})
            return self.tr_result
        else:
            self.sess.run(self.result, feed_dict={self.input: input})
            return self.result


class DQN():
    """DQNetwork with C51, Duelling, and NoisyNetwork

    """

    def __init__(self, args, action_space):
        self.action_space = action_space
        self.atoms = args.atoms

        self.inputs = tf.placeholder(tf.float32, [None, 84,84,args.history_length], name="inputs")
        # self.act = tf.placeholder(tf.float32, [None, self.action_size], name="act")

        filter1 = tf.Variable(tf.random_normal([8, 8, args.history_length, 32], stddev=0.01))
        self.conv1 = tf.nn.conv2d(input=self.inputs, filter=filter1, strides=[1,4,4,1], padding='SAME')
        self.h1 = tf.nn.relu(self.conv1)
        filter2 = tf.Variable(tf.random_normal([4,4,32,64], stddev=0.01))
        self.conv2 = tf.nn.conv2d(input=self.h1, filter=filter2, strides=[1,2,2,1], padding='VALID')
        self.h2 = tf.nn.relu(self.conv2)
        filter3 = tf.Variable(tf.random_normal([3,3,64,64], stddev=0.01))
        self.conv3 = tf.nn.conv2d(input=self.h2, filter=filter3, strides=[1,1,1,1], padding='VALID')
        self.h3 = tf.nn.relu(self.conv3)

        self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.init_noisy_std)
        self.r_v = self.fc_h_v.forward(self.h3)
        self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.init_noisy_std)
        self.r_a = self.fc_h_a.forward(self.h3)
        self.h_fc_h_v = tf.nn.relu(self.r_v)
        self.h_fc_h_a = tf.nn.relu(self.r_a)
        
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.init_noisy_std)
        self.z_v = self.fc_h_v.forward(self.h_fc_h_v)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.init_noisy_std)
        self.z_a = self.fc_h_v.forward(self.h_fc_h_a)
        self.h_fc_z_v = tf.nn.relu(self.z_v)
        self.h_fc_z_a = tf.nn.relu(self.z_a)

        self.q = self.h_fc_h_v + self.h_fc_h_v - tf.reduce_mean(self.h_fc_h_v, axis=1, keep_dims=True)
        self.action = tf.nn.softmax(self.q)
        self.action_log = tf.nn.log_softmax(self.q)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        
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

    def forward(self, inputs, log=False):
        if log:
            self.sess.run(self.action_log, feed_dict={self.inputs: inputs})
            return self.action_log
        else:
            self.sess.run(self.action, feed_dict={self.inputs: inputs})
            return self.action

    def reset_noise(self):
        self.sess.run(self.fc_h_v.reset_noise)
        self.sess.run(self.fc_h_a.reset_noise)
        self.sess.run(self.fc_z_v.reset_noise)
        self.sess.run(self.fc_z_a.reset_noise)
