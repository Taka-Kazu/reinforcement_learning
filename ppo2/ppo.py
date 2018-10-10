# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os

from tensorflow.python import debug as tf_debug

EP_MAX = 10000
EP_LEN = 200
GAMMA = 0.9
BATCH = 512
EPOCH = 1
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
CLIP_EPSILON = 0.2
NUM_HIDDENS = [256, 256, 256]
LEARNING_RATE = 1e-4
BETA = 1e-3# entropy
A_LR = 0.0001
C_LR = 0.0002
CELL_SIZE = 64

# directories
LOG_DIR = "./log"
MODEL_DIR = "./models"

MODEL_SAVE_INTERVAL = 100
RENDER_EP = 100

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

    self.global_step = tf.Variable(0., trainable=False)
    self.learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step, 1000, 0.98, staircase=True)

    with tf.name_scope("lstm"):
      # https://github.com/MatheusMRFM/A3C-LSTM-with-Tensorflow/blob/master/Network.py
      # https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_RNN.py
      lstm_input = tf.expand_dims(self.s_t, [0], name="lstm_input")
      step_size = tf.shape(self.s_t)[:1]
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, name="lstm_cell")
      c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
      h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
      self.lstm_init_op = [c_init, h_init]
      c_input = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name="c_input")
      h_input = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name="h_input")

      self.state_input = [c_input, h_input]
      lstm_state_input = tf.nn.rnn_cell.LSTMStateTuple(c_input, h_input)
      outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                               inputs=lstm_input,
                                               initial_state=lstm_state_input,
                                               sequence_length=step_size,
                                               time_major=False# batch_major
                                               )
      lstm_c, lstm_h = final_state
      self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
      self.cell_out = tf.reshape(outputs, [-1, CELL_SIZE], name="flatten_rnn_outputs")

    # common
    pi, pi_params = self._build_net('pi', trainable=True)
    oldpi, oldpi_params = self._build_net('oldpi', trainable=False)

    # critic
    with tf.variable_scope('critic'):
      lc = tf.layers.dense(self.cell_out, NUM_HIDDENS[0], tf.nn.relu, name="lc")
      self.v = tf.layers.dense(lc, 1, name="value")
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
      la0 = tf.layers.dense(self.cell_out, NUM_HIDDENS[0], tf.nn.relu, name="la0")
      la1 = tf.layers.dense(la0, NUM_HIDDENS[1], tf.nn.relu, name="la1")
      mu = tf.layers.dense(la1, NUM_ACTIONS, tf.nn.tanh, name="mu", trainable=trainable)
      sigma = tf.layers.dense(la1, NUM_ACTIONS, tf.nn.softplus, name="sigma", trainable=trainable)
      norm_dist = tf.distributions.Normal(loc=mu * A_BOUNDS[1], scale=sigma)
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return norm_dist, params

  def update(self, s, a, r, rnn_state_):
    for _ in range(EPOCH):
      self.sess.run(self.update_oldpi_op)
      adv = self.sess.run(self.advantage, {self.s_t: s, self.tfdc_r: r, self.state_input[0]: rnn_state_[0], self.state_input[1]: rnn_state_[1]})
      # adv = (adv - adv.mean())/(adv.std()+1e-6)   # sometimes helpful

      # update actor
      [self.sess.run(self.a_train_op, {self.s_t: s, self.a_t: a, self.adv: adv, self.state_input[0]: rnn_state_[0], self.state_input[1]: rnn_state_[1]}) for _ in range(A_UPDATE_STEPS)]

      # update critic
      [self.sess.run(self.c_train_op, {self.s_t: s, self.tfdc_r: r, self.state_input[0]: rnn_state_[0], self.state_input[1]: rnn_state_[1]}) for _ in range(C_UPDATE_STEPS)]

    _feed_dict={self.s_t:s, self.a_t:a, self.tfdc_r:r, self.adv:adv, self.state_input[0]: rnn_state_[0], self.state_input[1]: rnn_state_[1]}
    summary_str = self.sess.run(summary_ops, feed_dict={
      summary_vars[0]:r.mean(),
      summary_vars[1]:self.sess.run(self.entropy, _feed_dict) / BETA,
      summary_vars[2]:self.sess.run(self.learning_rate),
      summary_vars[3]:self.sess.run(self.a_loss, _feed_dict),
      summary_vars[4]:self.sess.run(self.c_loss, _feed_dict),
      summary_vars[5]:self.sess.run(self.v, _feed_dict).mean()
    })
    global GLOBAL_EP
    writer.add_summary(summary_str, global_step=GLOBAL_EP)
    writer.flush()

  def choose_action(self, s, cell_state):
    s = s[np.newaxis, :]
    a, cell_state_ = self.sess.run([self.sample_op, self.state_out], {self.s_t: s, self.state_input[0]:cell_state[0], self.state_input[1]:cell_state[1]})
    return np.clip(a[0], A_BOUNDS[0], A_BOUNDS[1]), cell_state_

  def get_v(self, s, rnn_state):
    if s.ndim < 2: s = s[np.newaxis, :]
    return self.sess.run(self.v, {self.s_t: s, self.state_input[0]: rnn_state[0], self.state_input[1]: rnn_state[1]})[0, 0]

env = gym.make('Pendulum-v0').unwrapped

NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]
A_BOUNDS = [env.action_space.low, env.action_space.high]
NONE_STATE = np.zeros(NUM_STATES)
GLOBAL_EP = 0

if __name__=="__main__":
  config = tf.ConfigProto(
    log_device_placement=True,
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(
      visible_device_list="0",
      allow_growth=True
    )
  )
  sess = tf.Session(config=config)
  #with tf.Session(config=config) as sess:
  with tf.device("/gpu:0"):
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
      rnn_state = ppo.lstm_init_op
      #keep_state = rnn_state.copy()

      for t in range(EP_LEN):    # in one episode
        if(ep % RENDER_EP == 0) and (ep != 0):
          env.render()
        a, rnn_state_ = ppo.choose_action(s, rnn_state)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)  # normalize reward, find to be useful
        s = s_
        rnn_state = rnn_state_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
          v_s_ = ppo.get_v(s_, rnn_state_)
          discounted_r = []
          for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
          discounted_r.reverse()

          bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
          buffer_s, buffer_a, buffer_r = [], [], []
          ppo.update(bs, ba, br, rnn_state_)
          #keep_state = rnn_state_.copy()
      print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r
      )
      GLOBAL_EP += 1
      if GLOBAL_EP % MODEL_SAVE_INTERVAL == 0:
        saver.save(sess, MODEL_DIR + "/ppo_model_ep_" + str(GLOBAL_EP) + ".ckpt")

