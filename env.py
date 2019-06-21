from collections import deque
import random
import atari_py
import tensorflow as tf
import numpy as np
import cv2


class Env():
  def __init__(self, args):
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0
    self.life_termination = False 
    self.window = args.history_length
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    state = state.astype(np.float32)
    return np.divide(state,255)
    # cv image - ndarray

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(np.zeros((84,84),dtype=np.float32))

  def reset(self):
    if self.life_termination:
      self.life_termination = False
      self.ale.act(0)
    else:
      self._reset_buffer()
      self.ale.reset_game()
      for _ in range(random.randrange(30)):
        self.ale.act(0)
        if self.ale.game_over():
          self.ale.reset_game()

    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return np.stack(list(self.state_buffer),0)

  def step(self, action):
    frame_buffer = np.zeros((2,84,84),dtype=np.float32)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)
    self.state_buffer.append(observation)
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:
        self.life_termination = not done
        done = True 
      self.lives = lives
    return np.stack(list(self.state_buffer), 0), reward, done

  def train(self):
    self.training = True

  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
