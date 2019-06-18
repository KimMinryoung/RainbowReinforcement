import os
import plotly
from plotly.graph_objs import Scatter, Line
import tensorflow as tf

from env import Env

# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10

# Test DQN
def test(args, T, dqn, val_mem, evaluate=False):
  global Ts, rewards, Qs, best_avg_reward
  env = Env(args)
  env.eval()
  Ts.append(T)
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, rewards_sum, done = env.reset(), 0, False

      action = dqn.act_e_greedy(state)    # Choose an action by ε-greedy method
      state, reward, done = env.step(action)
      rewards_sum += reward

      if args.render:
        env.render()

      if done:
        T_rewards.append(rewards_sum)
        break

  env.close()

  # Test action-value functions over validation memory
  for state in val_mem:
    T_Qs.append(dqn.evaluate_q(state))

  avg_reward, avg_Q = (sum(T_rewards) / len(T_rewards)), (sum(T_Qs) / len(T_Qs))

  if not evaluate:
    rewards.append(T_rewards)
    Qs.append(T_Qs)

    plot_line(Ts, rewards, 'Reward', path='results')
    plot_line(Ts, Qs, 'Q', path='results')

    if avg_reward > best_avg_reward:
      best_avg_reward = avg_reward
      dqn.save('results')

  return avg_reward, avg_Q

def plot_line(xs, ys_pop, title, path=''):
  color_a, color_b, color_c = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

  ys = tf.tensor(ys_pop, dtype=tf.float32)
  ys_max, ys_mean, ys_min, ys_std = ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.min(1)[0].squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = (ys_mean + ys_std), (ys_mean - ys_std)

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=color_a, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='Mean + STD', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=color_c, line=Line(color=color_b), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=color_c, line=Line(color='transparent'), name='Mean - STD', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=color_a, dash='dash'), name='Min')

  plotly.offline.plot({
    'data' : [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout' : dict(title=title, xaxis={'Title' : 'Step'}, yaxis={'Title' : title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)







