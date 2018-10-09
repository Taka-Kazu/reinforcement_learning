# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os

from tensorflow.python import debug as tf_debug

RENDER_EP = 100
EP_MAX = 10000
EP_LEN = 200
GAMMA = 0.9
BATCH = 512
EPOCH = 3
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
CLIP_EPSILON = 0.2
NUM_HIDDENS = [256, 256, 256]
LEARNING_RATE = 2e-4
BETA = 1e-3# entropy
A_LR = 0.0001
C_LR = 0.0002

# directories
LOG_DIR = "./log"
MODEL_DIR = "./models"

def build_summaries():
  with tf.name_scope("logger"):
    reward = tf.Variable(0.)
    tf.summary.scalar("Reward",reward)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy",entropy)
    learning_rate = tf.Variable(0.)
    tf.summary.scalar("Learning_Rate",learning_rate)
    policy_loss = tf.Variable(0.)
    tf.summary.scalar("Policy_Loss",policy_loss)
    value_loss = tf.Variable(0.)
    tf.summary.scalar("Value_Loss",value_loss)
    value_estimate = tf.Variable(0.)
    tf.summary.scalar("Value_Estimate",value_estimate)

    summary_vars = [reward,entropy,learning_rate,policy_loss,value_loss,value_estimate]
    #summary_vars = [reward,learning_rate,policy_loss,value_loss,value_estimate]
    summary_ops = tf.summary.merge_all()

  return summary_ops, summary_vars

class PPO(object):

  def __init__(self, sess):
    self.sess = sess
    self.s_t = tf.placeholder(tf.float32, [None, NUM_STATES], 'state')

    self.global_step = tf.Variable(0, trainable=False)
    self.learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step, 1000, 0.98, staircase=True)

    # common
    pi, self.v, pi_params = self._build_net('pi', trainable=True)
    oldpi, _, oldpi_params = self._build_net('oldpi', trainable=False)

    # critic
    with tf.variable_scope('critic'):
      self.tfdc_r = tf.placeholder(tf.float32, [None, 1], name='discounted_r')
      self.advantage = self.tfdc_r - self.v
      self.c_loss = tf.reduce_mean(tf.square(self.advantage))
      self.c_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.c_loss, global_step=self.global_step)

    # actor
    with tf.variable_scope('sample_action'):
      self.sample_op = tf.squeeze(pi.sample(1), axis=0)   # choosing action
    with tf.variable_scope('update_oldpi'):
      self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
    self.a_t = tf.placeholder(tf.float32, [None, NUM_ACTIONS], 'action')
    self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
    with tf.variable_scope('loss'):
      with tf.variable_scope('entropy'):
        self.entropy = BETA * tf.reduce_mean(pi.entropy())
      with tf.variable_scope('surrogate'):
        # ratio = tf.exp(pi.log_prob(self.a_t) - oldpi.log_prob(self.a_t))
        # ratio = pi.prob(self.a_t) / oldpi.prob(self.a_t)
        old_prob = oldpi.prob(self.a_t) + 1e-10
        ratio = pi.prob(self.a_t) / old_prob
        surr = ratio * self.adv
      self.a_loss = -tf.reduce_mean(tf.minimum(
        surr,
        tf.clip_by_value(ratio, 1.-CLIP_EPSILON, 1.+CLIP_EPSILON)*self.adv) - self.entropy)

    with tf.variable_scope('a_train'):
      self.a_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss, global_step=self.global_step)

  def _build_net(self, name, trainable):
    with tf.variable_scope(name):
      l0 = tf.layers.dense(self.s_t, NUM_HIDDENS[0], tf.nn.relu, name="l0")
      l1 = tf.layers.dense(l0, NUM_HIDDENS[1], tf.nn.relu, name="l1")
      mu = tf.layers.dense(l1, NUM_ACTIONS, tf.nn.tanh, name="mu", trainable=trainable)
      sigma = tf.layers.dense(l1, NUM_ACTIONS, tf.nn.softplus, name="sigma", trainable=trainable)
      norm_dist = tf.distributions.Normal(loc=mu * A_BOUNDS[1], scale=sigma)
      v = tf.layers.dense(l1, 1, name="value")
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return norm_dist, v, params

  def update(self, s, a, r):
    for _ in range(EPOCH):
      self.sess.run(self.update_oldpi_op)
      adv = self.sess.run(self.advantage, {self.s_t: s, self.tfdc_r: r})
      # adv = (adv - adv.mean())/(adv.std()+1e-6)   # sometimes helpful

      # update actor
      [self.sess.run(self.a_train_op, {self.s_t: s, self.a_t: a, self.adv: adv}) for _ in range(A_UPDATE_STEPS)]

      # update critic
      [self.sess.run(self.c_train_op, {self.s_t: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    _feed_dict={self.s_t:s, self.a_t:a, self.tfdc_r:r, self.adv:adv}
    summary_str = self.sess.run(summary_ops, feed_dict={
      summary_vars[0]:r.mean(),
      summary_vars[1]:self.sess.run(self.entropy, feed_dict={self.s_t:s}) / BETA,
      summary_vars[2]:self.sess.run(self.learning_rate),
      summary_vars[3]:self.sess.run(self.a_loss, _feed_dict),
      summary_vars[4]:self.sess.run(self.c_loss, feed_dict={self.s_t:s, self.tfdc_r:r}),
      summary_vars[5]:self.sess.run(self.v, feed_dict={self.s_t:s}).mean()
    })
    global GLOBAL_EP
    writer.add_summary(summary_str, global_step=GLOBAL_EP)
    writer.flush()

  def choose_action(self, s):
    s = s[np.newaxis, :]
    a = self.sess.run(self.sample_op, {self.s_t: s})[0]
    return np.clip(a, A_BOUNDS[0], A_BOUNDS[1])

  def get_v(self, s):
    if s.ndim < 2: s = s[np.newaxis, :]
    return self.sess.run(self.v, {self.s_t: s})[0, 0]

env = gym.make('Pendulum-v0').unwrapped

NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]
A_BOUNDS = [env.action_space.low, env.action_space.high]
NONE_STATE = np.zeros(NUM_STATES)
GLOBAL_EP = 0

if __name__=="__main__":
  with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    ppo = PPO(sess)
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    saver = tf.train.Saver()

    for ep in range(EP_MAX):
      s = env.reset()
      buffer_s, buffer_a, buffer_r = [], [], []
      ep_r = 0
      for t in range(EP_LEN):    # in one episode
        if(ep % RENDER_EP == 0) and (ep != 0):
          env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)  # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
          v_s_ = ppo.get_v(s_)
          discounted_r = []
          for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
          discounted_r.reverse()

          bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
          buffer_s, buffer_a, buffer_r = [], [], []
          ppo.update(bs, ba, br)
      print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        "|GLOBAL_EP: %i" % GLOBAL_EP
      )
      GLOBAL_EP += 1

