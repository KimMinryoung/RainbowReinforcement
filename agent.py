import random
import tensorflow as tf
import numpy as np
from model import DQN

class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = np.linspace(args.V_min, args.V_max, self.atoms)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_steps
    self.discount = args.discount
    self.norm_clip = args.max_norm_clip
    
    self.sess = tf.Session()

    with tf.variable_scope("online_net"):
      self.online_net = DQN(args, self.action_space)
    
    self.online_net.train()

    with tf.variable_scope("target_net"):
      self.target_net = DQN(args, self.action_space)
    self.target_net.train()
    
    self.sess.run(tf.global_variables_initializer())
    
    self.saver = tf.train.Saver()
    if tf.gfile.Exists("./models/model.ckpt"):
      self.saver.restore(self.sess, "./models/model.ckpt")
    
    online_net_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="online_net")
    target_net_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
    update_target_op = []
    for var, var_target in zip(sorted(online_net_func_vars, key=lambda v: v.name),
                               sorted(target_net_func_vars, key=lambda v: v.name)):
        update_target_op.append(var_target.assign(var))
    self.update_target_op = tf.group(*update_target_op)
    
    self.update_target_net()

    self.optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate, epsilon = args.adam_eps)

  def forward(self, network, inputs,  log=False):
    if log:
      output = self.sess.run(network.action_log, feed_dict={network.inputs: inputs})
      return output
    else:
      output = self.sess.run(network.action, feed_dict={network.inputs: inputs})
      return output
    
  def reset_noise(self):
    self.online_net.reset_noise()

  def act(self, state):
    return np.argmax(np.sum((self.forward(self.online_net, state.reshape(1,84,84,4)) * self.support), axis=-1))
  
  def act_e_greedy(self, state, epsilon=0.001):
    return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

  def learn(self, mem):
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    log_ps = self.forward(self.online_net, states, log=True)
    log_ps_a = []
    for i in range(self.batch_size):
      log_ps_a.append(log_ps[i][actions[i]])

    pns = self.forward(self.online_net, next_states)
    dns = np.broadcast_to(self.support, (self.action_space, self.atoms))
    dns = np.multiply(np.broadcast_to(dns, (self.batch_size, self.action_space, self.atoms)), pns )
    argmax_indices_ns = np.argmax(np.sum(dns, axis=2), axis=1)
    self.target_net.reset_noise()
    pns = self.forward(self.target_net, next_states)
    pns_a = pns[range(self.batch_size), argmax_indices_ns]
    
    Tz = np.expand_dims(returns, axis=1) + (self.discount ** self.n) * np.multiply(nonterminals,np.expand_dims(self.support,axis=0))
    Tz = np.clip(Tz, self.Vmin, self.Vmax)
    b = (Tz - self.Vmin) / self.delta_z
    l, u = np.floor(b).astype(dtype=np.int64) , np.ceil(b).astype(dtype=np.int64)
    l[(u > 0) * (l == u)] -= 1
    u[(l < (self.atoms - 1)) * (l == u)] += 1

    m = np.zeros([self.batch_size,self.atoms], dtype=states.dtype)
    offset = np.broadcast_to(np.expand_dims(np.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size),axis=1),(self.batch_size,self.atoms)).astype(actions.dtype)
    np.add.at(m.flatten(), (l + offset).flatten(), (pns_a * (u.astype(np.float32) - b)).flatten())
    np.add.at(m.flatten(), (u + offset).flatten(), (pns_a * (b - l.astype(np.float32))).flatten())

    loss = -np.sum(m * log_ps_a, 1)
    loss = weights * loss
    
    #tvars = tf.trainable_variables()
    #grads, _ = tf.clip_by_norm(tvars, self.norm_clip)
    #train = self.optimizer.apply_gradients(zip(grads, tvars))
    
    #train = self.optimizer.minimize(loss)
    #self.sess.run(train)
    
    #tf.clip_by_norm(self.online_net.parameters(), self.norm_clip)

    #mem.update_priorities(idxs, loss.detach())

  def update_target_net(self):
    self.sess.run(self.update_target_op)

  def save(self, path):
    self.save_path = self.saver.save(self.sess, "./models/model.ckpt")

  def evaluate_q(self, state):
    return np.sum((self.forward(self.online_net, state.reshape(1,84,84,4)) * self.support), axis=-1).max(axis=1)[0]

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
