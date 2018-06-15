#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import gym
import tensorflow as tf
from collections import deque
import os

episodes = 300
steps = 200

target_step = 195

env = gym.make("CartPole-v0")
# actionは0(左)と1(右)の2次元
# observationは0(カート位置),1(カート速度),2(ポール角度),3(ポール速度)の4次元連続値
# rewardは1stepごとに1

class DQN:
  def __init__(self, state_dim, action_dim):
    self.action_dim = action_dim
    self.state_dim = state_dim
    self.hidden_dim = 32
    self.learning_rate = 0.001
    self.minibatch_size = 32
    self.replay_memory_size = 500
    self.gamma = 0.99 # 割引率
    self.memory = deque(maxlen = self.replay_memory_size)
    self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    self.model_name = "model.ckpt"
    self.init_model()

  def init_model(self):
    # 入力層
    self.x = tf.placeholder(tf.float32, [None, self.state_dim])
    # 隠れ層
    w_fc1 = tf.Variable(tf.truncated_normal([self.state_dim, self.hidden_dim], stddev=0.01))
    b_fc1 = tf.Variable(tf.zeros([self.hidden_dim]))
    h_fc1 = tf.nn.relu(tf.matmul(self.x, w_fc1) + b_fc1)
    # 出力層
    w_out = tf.Variable(tf.truncated_normal([self.hidden_dim, self.action_dim], stddev=0.01))
    b_out = tf.Variable(tf.zeros([self.action_dim]))
    #self.y = tf.nn.softmax(tf.matmul(h_fc1, w_out) + b_out)
    self.y = tf.matmul(h_fc1, w_out) + b_out

    # 学習用
    self.y_ = tf.placeholder(tf.float32, [None, self.action_dim])
    #self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))
    self.loss = tf.losses.huber_loss(self.y_, self.y)

    optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
    self.training = optimizer.minimize(self.loss)

    self.saver = tf.train.Saver()

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def get_action(self, state_t, episode, learning = True):
    if(learning):
      epsilon = 0.001 + 0.9 / (1.0 + episode)
      if epsilon <= np.random.uniform(0, 1):
        y = self.sess.run(self.y, feed_dict={self.x:state_t})
        action = np.argmax(y)
      else:
        action = np.random.randint(self.action_dim)
    else:
      y = self.sess.run(self.y, feed_dict={self.x:state_t})
      action = np.argmax(y)

    return action

  def experience_replay(self):
    state_minibatch = []
    y_minibatch = []

    minibatch_size = min(len(self.memory), self.minibatch_size)
    minibatch_indexes = np.random.randint(0, len(self.memory), minibatch_size)

    for j in minibatch_indexes:
      state_t, state_t_1, reward, terminal, action = self.memory[j]
      #print self.memory[j]
      # Q値
      _y = self.sess.run(self.y, feed_dict={self.x:state_t})[0]
      #print "---_y---"
      #print _y
      if terminal:
        _y[action] = reward
      else:
        max_q =  np.max(self.sess.run(self.y, feed_dict={self.x:state_t_1}))
        _y[action] = reward + self.gamma * max_q
      state_minibatch.append(state_t[0])
      y_minibatch.append(_y)

    #print "---state_minibatch---"
    #print state_minibatch
    #print type(state_minibatch[0])
    #print "---y_minibatch---"
    #print y_minibatch
    #print type(y_minibatch[0])
    self.sess.run(self.training, feed_dict={self.x:state_minibatch, self.y_:y_minibatch})
    #self.current_loss = self.sess.run(self.loss, feed_dict={self.x:state_minibatch, self.y_:y_minibatch})
    #print self.current_loss

  def store_experience(self, state_t, state_t_1, reward, terminal, action):
    self.memory.append((state_t, state_t_1, reward, terminal, action))

print "--- learning start ---"

agent = DQN(4, 2)

for episode in range(episodes):
  observation = env.reset()
  state_t_1 = observation.reshape((1, observation.size))
  state_t = state_t_1
  action = agent.get_action(state_t, episode)
  for step in range(steps):
    #env.render()
    state_t = state_t_1
    action = agent.get_action(state_t, episode)
    observation, reward, terminal, info = env.step(action)
    state_t_1 = observation.reshape((1, observation.size))
    if terminal:
      if step > target_step:
        reward = 1
      else:
        reward = -1
    else:
      reward = 0
    agent.store_experience(state_t, state_t_1, reward, terminal, action)
    if agent.minibatch_size < len(agent.memory):
      agent.experience_replay()
    if terminal:
      print "episode:", episode
      print "finished at", step, "step"
      break

print "--- learning result ---"
observation = env.reset()
state_t_1 = observation.reshape((1, observation.size))
state_t = state_t_1
action = agent.get_action(state_t, episode, False)
for step in range(steps):
  env.render()
  state_t = state_t_1
  action = agent.get_action(state_t, episode, False)
  observation, reward, terminal, info = env.step(action)
  state_t_1 = observation.reshape((1, observation.size))
  if terminal:
    print "finished at", step, "step"
    break

env.close()

