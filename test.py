import os
import plotly
from plotly.graph_objs import Scatter, Line
import tensorflow as tf
import numpy as np
import warnings

from env import Env

# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10

# Plot
def save_graph(xs, ys_pop, title, path=''):
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  
  color_a, color_b, color_c = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

  ys = np.asarray(ys_pop, dtype=np.float32)
  ys_max = ys.max(axis=1)
  ys_mean = ys.mean(axis=1)
  ys_min = ys.min(axis=1)
  ys_std = ys.std(axis=1)
  ys_upper, ys_lower = (ys_mean + ys_std), (ys_mean - ys_std)

  trace_max = Scatter(x=xs, y=ys_max, line=Line(color=color_a, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color='lightblue'), name='Mean + STD', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=color_c, line=Line(color=color_b), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=color_c, line=Line(color='lightpink'), name='Mean - STD', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min, line=Line(color=color_a, dash='dash'), name='Min')

  plotly.offline.plot({
    'data' : [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout' : dict(title=title, xaxis={'title' : 'Step'}, yaxis={'title' : title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)

def test(args, T, dqn_agent, valid_memory, evaluate=False):
  global Ts, rewards, Qs, best_avg_reward
  env = Env(args)
  env.eval()
  Ts.append(T)
  T_rewards, T_Qs = [], []
  done = True
  
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, rewards_sum, done = env.reset(), 0, False

      action = dqn_agent.act_e_greedy(state)    # Choose an action by ε-greedy method
      state, reward, done = env.step(action)
      rewards_sum += reward

      if args.render:
        env.render()

      if done:
        T_rewards.append(rewards_sum)
        break

  env.close()

  # Test action-value functions over validation memory
  for state in valid_memory:
    T_Qs.append(dqn_agent.evaluate_q(state))

  avg_reward, avg_Q = (sum(T_rewards) / len(T_rewards)), (sum(T_Qs) / len(T_Qs))

  if not evaluate:
    rewards.append(T_rewards)
    Qs.append(T_Qs)

    save_graph(Ts, rewards, 'Reward', path='results')
    save_graph(Ts, Qs, 'Q', path='results')

    if avg_reward > best_avg_reward:
      best_avg_reward = avg_reward
      dqn_agent.save('results')

  return avg_reward, avg_Q


