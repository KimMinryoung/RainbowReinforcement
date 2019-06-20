import random
import tensorflow as tf
import numpy as np
from model_yjs import DQN

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

    #self.online_net = DQN(args, self.action_space).to(device=args.device)
    with tf.variable_scope("online_net"):
      self.online_net = DQN(args, self.action_space)
    
    #if args.model and os.path.isfile(args.model):
    #  self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))
    self.online_net.train()

    #self.target_net = DQN(args, self.action_space).to(device=args.device)
    with tf.variable_scope("target_net"):
      self.target_net = DQN(args, self.action_space)
    self.target_net.train()
    #for param in self.target_net.parameters():
    #  param.requires_grad = False
    
    self.sess.run(tf.global_variables_initializer())
    
    self.saver = tf.train.Saver()
    if tf.gfile.Exists("./models/model.ckpt"):
      self.saver.restore(self.sess, "./models/model.ckpt")
    
    # make an op for target update
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
          
  # Acts based on single state (no batch)
  def act(self, state):
    #return (self.forward(self.online_net, state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
    return np.argmax(np.sum((self.forward(self.online_net, state.reshape(1,84,84,4)) * self.support), axis=-1))
  

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.forward(self.online_net, states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    #print("log_ps.shape: " + str(log_ps.shape))
    #print("actions.shape: " + str(actions.shape))
    #log_ps_a = [np.ndarray([self.batch_size, self.atoms],dtype=np.float32)]
    log_ps_a = []
    for i in range(self.batch_size):
      log_ps_a.append(log_ps[i][actions[i]])
    #log_ps_a = log_ps[0:, actions]  # log p(s_t, a_t; θonline)

    pns = self.forward(self.online_net, next_states)
   # print("pns shape : " + str(pns.shape))
    dns = np.broadcast_to(self.support, (self.action_space, self.atoms))
    dns = np.multiply(np.broadcast_to(dns, (self.batch_size, self.action_space, self.atoms)), pns )
    #print("dns shape : " + str(dns.shape))
    argmax_indices_ns = np.argmax(np.sum(dns, axis=2), axis=1)
    #sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
    self.target_net.reset_noise()  # Sample new target net noise
    pns = self.forward(self.target_net, next_states)  # Probabilities p(s_t+n, ·; θtarget)
    pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

    # Compute Tz (Bellman operator T applied to z)
    #Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
    #Tz = returns.reshape(unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
    Tz = np.expand_dims(returns, axis=1) + (self.discount ** self.n) * np.multiply(nonterminals,np.expand_dims(self.support,axis=0))
    #Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
    Tz = np.clip(Tz, self.Vmin, self.Vmax)
    # Compute L2 projection of Tz onto fixed support z
    b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
    #l, u = tf.cast(b.floor(), tf.int64), tf.cast(b.ceil(), tf.int64)
    l, u = np.floor(b).astype(dtype=np.int64) , np.ceil(b).astype(dtype=np.int64)
    # Fix disappearing probability mass when l = b = u (b is int)
    l[(u > 0) * (l == u)] -= 1
    u[(l < (self.atoms - 1)) * (l == u)] += 1

    # Distribute probability of Tz
    #m = states.new_zeros(self.batch_size, self.atoms)
    #print(states.dtype)
    m = np.zeros([self.batch_size,self.atoms], dtype=states.dtype)
    #offset = tf.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
    offset = np.broadcast_to(np.expand_dims(np.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size),axis=1),(self.batch_size,self.atoms)).astype(actions.dtype)
    #m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
    #m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
    np.add.at(m.flatten(), (l + offset).flatten(), (pns_a * (u.astype(np.float32) - b)).flatten())
    np.add.at(m.flatten(), (u + offset).flatten(), (pns_a * (b - l.astype(np.float32))).flatten())

    loss = -np.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    loss = weights * loss  # Importance weight losses before prioritised experience replay (done after for original/non-distributional version)
    
    #tvars = tf.trainable_variables()
    #grads, _ = tf.clip_by_norm(tvars, self.norm_clip)
    #train = self.optimizer.apply_gradients(zip(grads, tvars))
    
    #train = self.optimizer.minimize(loss)
    #self.sess.run(train)
    
    #tf.clip_by_norm(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    #nn.utils.clip_grad_norm_(self.online_net.parameters(), self.norm_clip)

    #mem.update_priorities(idxs, loss.detach())  # Update priorities of sampled transitions

  def update_target_net(self): # ?!?!?!?!?!?!?!
    self.sess.run(self.update_target_op)
    
    '''e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
      op = e2_v.assign(e1_v)
      update_ops.append(op)
    sess.run(update_ops)'''
    
    #self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path):
    self.save_path = self.saver.save(self.sess, "./models/model.ckpt")
    #torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    return np.sum((self.forward(self.online_net, state.reshape(1,84,84,4)) * self.support), axis=-1).max(axis=1)[0]
    #return (self.forward(self.online_net, state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
