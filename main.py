import argparse
import random
import tensorflow as tf

from agent import Agent
from datetime import datetime
from env import Env
from memory import ReplayMemory
from test import test

# Argument parsing
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--game', type=str, default='space_invaders', help='game title')
parser.add_argument('--max-step', type=int, default=int(50e6), metavar='STEPS', help='Maximum training steps (number of frames x 4)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LEN', help='Maximum episode length (set to 0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='NUM', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Size of hidden')
parser.add_argument('--init-noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--atoms', type=int, default=51, metavar='SIZE', help='Discretized size of value distribution')
parser.add_argument('--V-min', type=float, default=-10.0, metavar='Vmin', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10.0, metavar='Vmax', help='Maximum of value distribution support')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAP', help='Capacity of experience replay memory')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='Fs', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritized experience replay exponent')
parser.add_argument('--init-priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritized experience replay important sampling weight')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discounting factor')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-duration', type=int, default=500, metavar='NUM', help='Number of transitions for evaluation')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='NUM', help='Number of evaluation episodes')
parser.add_argument('--evaluation-interval', type=int, default=1000, metavar='STEPS', help='Training steps between evaluations')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--log-interval', type=int, default=100, metavar='STEPS', help='Training steps between logging status')
parser.add_argument('--max-norm-clip', type=float, default=10, metavar='NORM', help='Maximum L2 norm for gradient clipping')
parser.add_argument('--multi-steps', type=int, default=3, metavar='STEPS', help='Steps for multi-step return')
parser.add_argument('--render', action='store_true', help='Display screen (for testing only)')
parser.add_argument('--reward-clip', type=int, default=1, metavar='R', help='Reward clipping (set to 0 to disable)')
parser.add_argument('--steps-before-train', type=int, default=int(50e3), metavar='STEPS', help='Steps before starting training')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ', help='Steps to update target network')

# Setup
args = parser.parse_args()
print(' ' * 26 + 'options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ' = ' + str(v))
random.seed(args.seed)
tf.random.set_random_seed(random.randint(1, 10000))
args.device = tf.device("/gpu:0")

def log(s):
    print(str(datetime.now().strftime('%Y.%m.%d. %H:%M:%S  ')) + s)

env = Env(args)
env.train()
action_space = env.action_space()

dqn = Agent(args, env)
mem = ReplayMemory(args, args.memory_capacity)
priority_weight_increase = (1 - args.init_priority_weight) / (args.max_step - args.steps_before_train)

val_mem = ReplayMemory(args, args.evaluation_duration)
step = 0
done = True
while step < args.evaluation_duration:
    if done:
        state, done = env.reset(), False

    next_state, _, done = env.step(random.randint(0, action_space - 1))
    val_mem.append(state, None, None, done)
    state = next_state
    step += 1

if args.evaluate:
    dqn.eval()
    avg_reward, avg_Q = test(args, 0, dqn, val_mem, evaluate=True)
    print('Average reward: ' + str(avg_reward) + '. Average Q: ' + str(avg_Q)+'.')
else:
    dqn.train()
    step = 0
    done = True
    while step < args.max_step:
        if done:
            state, done = env.reset(), False

        if (step % args.replay_frequency) == 0:
            dqn.reset_noise()

        action = dqn.act(state)
        next_state, reward, done = env.step(action)

        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)

        mem.append(state, action, reward, done)
        step += 1

        if (step % args.log_interval) == 0:
            log('step: ' + str(step))

        if step >= args.steps_before_train:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            if (step % args.replay_frequency) == 0:
                dqn.learn(mem)

            if (step % args.evaluation_interval) == 0:
                dqn.eval()
                avg_reward, avg_Q = test(args, step, dqn, val_mem)
                log('step: ' + str(step) + '. Average reward: ' + str(avg_reward) + '. Average Q: ' + str(avg_Q)+'.')
                dqn.train()

            if (step % args.target_update) == 0:
                dqn.update_target_net()

        state = next_state

env.close()

























