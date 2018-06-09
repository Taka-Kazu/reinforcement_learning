#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import gym

episodes = 1000
steps = 200

target_step = 195

env = gym.make("CartPole-v0")
# actionは0(左)と1(右)の2次元
# observationは0(カート位置),1(カート速度),2(ポール角度),3(ポール速度)の4次元連続値
# rewardは1stepごとに1

class QLearning:
  def __init__(self, state_dim, action_dim):
    self.digitize_state_num = 6 # stateの離散化
    self.action_dim = action_dim
    self.q_table = np.random.uniform(low = -1, high = 1, size=(state_dim ** self.digitize_state_num, action_dim))

  def digitize_state(self, observation):
    #離散化
    #[1:-1]は最初と最後の要素以外のスライス
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized_observation = [
      np.digitize(cart_pos, bins = np.linspace(-2.4, 2.4, self.digitize_state_num + 1)[1:-1]),
      np.digitize(cart_v, bins = np.linspace(-3.0, 3.0, self.digitize_state_num + 1)[1:-1]),
      np.digitize(pole_angle, bins = np.linspace(-0.5, 0.5, self.digitize_state_num + 1)[1:-1]),
      np.digitize(pole_v, bins = np.linspace(-2.0, 2.0, self.digitize_state_num + 1)[1:-1])]
    return sum([x * (self.digitize_state_num ** i) for i, x in enumerate(digitized_observation)])

  def get_action(self, state_t, episode, learning = True):
    if(learning):
      epsilon = 0.5 * (0.99 ** episode)
      if epsilon <= np.random.uniform(0, 1):
        action = np.argmax(self.q_table[state_t])
      else:
        action = np.random.randint(self.action_dim)
    else:
      action = np.argmax(self.q_table[state_t])

    return action

  def update(self, state_t, state_t_1, reward, terminal, action):
    gamma = 0.99 # 割引率
    alpha = 0.5 # 学習率
    q_max = np.argmax(self.q_table[state_t_1])
    q_max = self.q_table[state_t_1, q_max]
    self.q_table[state_t, action] = (1 - alpha) * self.q_table[state_t, action] + alpha * (reward + gamma * q_max)

agent = QLearning(4, 2)

for episode in range(episodes):
  observation = env.reset()
  state_t_1 = agent.digitize_state(observation)
  state_t = state_t_1
  action = agent.get_action(state_t, episode)
  for step in range(steps):
    #env.render()
    observation, reward, terminal, info = env.step(action)
    state_t = state_t_1
    state_t_1 = agent.digitize_state(observation)
    action = agent.get_action(state_t, episode)
    if terminal:
      reward = -200
    agent.update(state_t, state_t_1, reward, terminal, action)
    if terminal:
      print "episode:", episode
      print "finished at", step, "step"
      break

print "--- learning result ---"
observation = env.reset()
state_t_1 = agent.digitize_state(observation)
state_t = state_t_1
action = agent.get_action(state_t, episode, False)
for step in range(steps):
  env.render()
  observation, reward, terminal, info = env.step(action)
  state_t = state_t_1
  state_t_1 = agent.digitize_state(observation)
  action = agent.get_action(state_t, episode, False)
  if terminal:
    print "finished at", step, "step"
    break

env.close()

