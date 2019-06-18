import math
import tensorflow as tf


class NoisyLinear():
    def __init__(self, in_, out_, std_init=0.5, training=True):
        self.in_ = in_
        self.out_ = out_
        self.std_init = std_init

        self.input = tf.placeholder(tf.float32, [None, self.in_], name="input")

        self.weight_mu = tf.Variable(tf.zeros([out_, in_]), name="weight_mu")
        self.weight_sigma = tf.Variable(tf.zeros([out_, in_]), name="weight_sigma")

        self.weight_epsilon = tf.constant

        self.bias_mu = tf.Variable(tf.zeros([out_]), name="bias_mu")
        self.bias_sigma = tf.Variable(tf.zeros([out_]), name="bias_sigma")

        self.bias_epsilon = tf.constant

        self.result = tf.matmul(self.input, self.weight_mu) + self.bias_mu
        self.tr_result = tf.matmul(self.input,
                                      self.weight_mu + self.weight_sigma * self.weight_epsilon) + self.bias_mu + self.bias_sigma * self.bias_epsilon

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

        self.inputs = tf.placeholder(tf.float32, [None, args.history_length * 8 * 32], name="inputs")
        # self.act = tf.placeholder(tf.float32, [None, self.action_size], name="act")

        self.conv1 = tf.nn.conv2d(input=self.inputs, filter=[8, 8, args.history_length, 32], strides=4, padding='SAME')
        self.h1 = tf.nn.relu(self.conv1)
        self.conv2 = tf.nn.conv2d(input=self.h1, filter=[4, 4, 32, 64], strides=2, padding='VALID')
        self.h2 = tf.nn.relu(self.conv2)
        self.conv3 = tf.nn.conv2d(input=self.h2, filter=[3, 3, 64, 64], strides=1, padding='VALID')
        self.h3 = tf.nn.relu(self.conv3)

        self.fc_h_v = NoisyLinear(self.inputs, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.inputs, args.hidden_size, std_init=args.noisy_std)
        self.h_fc_h_v = tf.nn.relu(self.fc_h_v)
        self.h_fc_h_a = tf.nn.relu(self.fc_h_a)
        self.fc_z_v = NoisyLinear(self.h_fc_h_v, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(self.h_fc_h_a, action_space * self.atoms, std_init=args.noisy_std)
        self.h_fc_z_v = tf.nn.relu(self.fc_z_v)
        self.h_fc_z_a = tf.nn.relu(self.fc_z_a)

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
