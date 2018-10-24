# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import threading, time

from tensorflow.python import debug as tf_debug

#import pyximport
#pyximport.install(inplace=True)
import myenv

GAME='Pendulum-v0'
#GAME='myenv-v2'
env = gym.make(GAME)
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]
A_BOUNDS = [env.action_space.low, env.action_space.high]
NONE_STATE = np.zeros(NUM_STATES)

EP_MAX = 10000
EP_LEN = 200
GAMMA = 0.99
LAMBDA = 0.95
NUM_WORKERS = 16
BATCH_SIZE = 512
EPOCH = 10
BUFFER_SIZE = BATCH_SIZE * EPOCH
TIME_HORIZON = BATCH_SIZE * EPOCH
CLIP_EPSILON = 0.2
NUM_HIDDENS = [256, 256, 256]
#LEARNING_RATE = 1e-4
LEARNING_RATE = 2e-5
BETA = 1e-4# entropy
VALUE_FACTOR = 1.0

# directories
LOG_DIR = "./log"
MODEL_DIR = "./models"

MODEL_SAVE_INTERVAL = 100
RENDER_EP = 100

GLOBAL_EP = 0
NN_MODEL = "/home/amsl/reinforcement_learning/ppo/models/ppo_model_ep_" + str(GLOBAL_EP) + ".ckpt"
if GLOBAL_EP == 0:
  NN_MODEL = None

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
    loss = tf.Variable(0.)
    tf.summary.scalar("Loss",loss)

    summary_vars = [reward,entropy,learning_rate,policy_loss,value_loss,value_estimate, loss]
    summary_ops = tf.summary.merge_all()

  return summary_ops, summary_vars

class PPO(object):

  def __init__(self, sess):
    self.sess = sess
    self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES), name='state')

    self.global_step = tf.Variable(0., trainable=False)
    #self.learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step, 1000, 0.98, staircase=True)
    self.learning_rate = LEARNING_RATE

    self.weight_init = tf.random_normal_initializer(0.0, 0.1)
    # common
    pi, pi_params, self.v = self._build_net('pi', trainable=True)
    oldpi, oldpi_params, _ = self._build_net('oldpi', trainable=False)

    # critic
    with tf.variable_scope('critic'):
      #lc = tf.layers.dense(self.s_t, NUM_HIDDENS[0], tf.nn.relu, name="lc")
      self.tfdc_r = tf.placeholder(tf.float32, [None, 1], name='discounted_r')
      self.c_loss = tf.reduce_mean(tf.square(self.tfdc_r - self.v))
      #self.c_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.c_loss)

    # actor
    with tf.variable_scope('sample_action'):
      self.sample_op = tf.squeeze(pi.sample(1), axis=0)   # choosing action
    with tf.variable_scope('update_oldpi'):
      self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
    self.a_t = tf.placeholder(tf.float32, [None, NUM_ACTIONS], 'action')
    self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
    with tf.variable_scope('loss'):
      with tf.variable_scope('entropy'):
        self.entropy = tf.reduce_mean(pi.entropy())
      with tf.variable_scope('surrogate'):
        # ratio = tf.exp(pi.log_prob(self.a_t) - oldpi.log_prob(self.a_t))
        # ratio = pi.prob(self.a_t) / oldpi.prob(self.a_t)
        old_prob = oldpi.prob(self.a_t) + 1e-10
        ratio = pi.prob(self.a_t) / old_prob
        surr = ratio * self.adv
      self.a_loss = -tf.reduce_mean(tf.minimum(
        surr,
        tf.clip_by_value(ratio, 1.-CLIP_EPSILON, 1.+CLIP_EPSILON)*self.adv))
      self.loss = self.a_loss - BETA * self.entropy + self.c_loss * VALUE_FACTOR

    with tf.variable_scope('train'):
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    self.brain_buffer_s = np.empty((0, NUM_STATES))
    self.brain_buffer_a = np.empty((0, NUM_ACTIONS))
    self.brain_buffer_r = np.empty((0, 1))
    self.brain_buffer_adv = np.empty((0, 1))


  def _build_net(self, name, trainable):
    with tf.variable_scope(name):
      la0 = tf.layers.dense(self.s_t, NUM_HIDDENS[0], tf.nn.relu, kernel_initializer=self.weight_init, name="la0")
      la1 = tf.layers.dense(la0, NUM_HIDDENS[1], tf.nn.relu, kernel_initializer=self.weight_init, name="la1")
      mu = tf.layers.dense(la1, NUM_ACTIONS, tf.nn.tanh, kernel_initializer=self.weight_init, name="mu", trainable=trainable)
      sigma = tf.layers.dense(la1, NUM_ACTIONS, tf.nn.softplus, kernel_initializer=self.weight_init, name="sigma", trainable=trainable)
      norm_dist = tf.distributions.Normal(loc=mu * A_BOUNDS[1], scale=sigma+1e-8)
      #prob = tf.layers.dense(la1, NUM_ACTIONS, tf.nn.softmax, kernel_initializer=self.weight_init, name="prob", trainable=trainable)
      v = tf.layers.dense(la1, 1, kernel_initializer=self.weight_init, name="value")
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return norm_dist, params, v

  def update(self):
    #print self.brain_buffer_s.shape
    #print self.brain_buffer_a.shape
    #print self.brain_buffer_r.shape
    #print self.brain_buffer_adv.shape
    for _ in range(EPOCH):
      for i in range(len(self.brain_buffer_s) // BATCH_SIZE):
        start = i * BATCH_SIZE
        end = (i+1) * BATCH_SIZE
        self.sess.run(self.update_oldpi_op)
        self.sess.run(self.train_op, {self.s_t: self.brain_buffer_s, self.a_t: self.brain_buffer_a, self.adv: self.brain_buffer_adv, self.tfdc_r: self.brain_buffer_r})
    self.brain_buffer_s = np.empty((0, NUM_STATES))
    self.brain_buffer_a = np.empty((0, NUM_ACTIONS))
    self.brain_buffer_r = np.empty((0, 1))
    self.brain_buffer_adv = np.empty((0, 1))


  def choose_action(self, s):
    s = s[np.newaxis, :]
    a = self.sess.run([self.sample_op], {self.s_t: s})
    return np.clip(a, A_BOUNDS[0], A_BOUNDS[1]).reshape(-1)

  def get_v(self, s):
    if s.ndim < 2: s = s[np.newaxis, :]
    return self.sess.run(self.v, {self.s_t: s})[0, 0]

  def push_buffer(self, s, a, r, adv):
    self.brain_buffer_s = np.append(self.brain_buffer_s, s, 0)
    self.brain_buffer_a = np.append(self.brain_buffer_a, a, 0)
    self.brain_buffer_r = np.append(self.brain_buffer_r, r, 0)
    self.brain_buffer_adv = np.append(self.brain_buffer_adv, adv, 0)
    _feed_dict={self.s_t:s, self.a_t:a, self.tfdc_r:r, self.adv:adv}
    summary_str = self.sess.run(summary_ops, feed_dict={
      summary_vars[0]:r.mean(),
      summary_vars[1]:self.sess.run(self.entropy, _feed_dict),
      #summary_vars[2]:self.sess.run(self.learning_rate),
      summary_vars[2]:(self.learning_rate),
      summary_vars[3]:self.sess.run(self.a_loss, _feed_dict),
      summary_vars[4]:self.sess.run(self.c_loss, _feed_dict),
      summary_vars[5]:self.sess.run(self.v, _feed_dict).mean(),
      summary_vars[6]:self.sess.run(self.loss, _feed_dict)
    })
    global GLOBAL_EP
    writer.add_summary(summary_str, global_step=GLOBAL_EP)
    writer.flush()


class Worker:
  def __init__(self, name, brain):
    self.env = gym.make(GAME)
    self.name = name

  def run(self):
    global GLOBAL_EP
    while not COORD.should_stop():
      s = self.env.reset()
      buffer_s, buffer_a, buffer_r = [], [], []
      buffer_v = []
      ep_r = 0
      for t in range(EP_LEN):    # in one episode
        #if(self.name=="W_0"):
        #  self.env.render()
        a = ppo.choose_action(s)
        start_time = time.time()
        s_, r, done, _ = self.env.step(a)
        elapsed_time = time.time() - start_time
        #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)  # normalize reward, find to be useful
        #buffer_r.append(r)
        buffer_v.append(ppo.get_v(s))
        s = s_
        ep_r += r

        # calcurate advantage
        if (t+1) % BATCH_SIZE == 0 or t == EP_LEN-1 or done:
          v_s_ = 0
          if not done:
            v_s_ = ppo.get_v(s_)
          buffer_v.append(v_s_)
          running_adv = 0.0
          advantage = []
          len_r = len(buffer_r)
          for t in reversed(range(len_r)):
            delta_t = buffer_r[t] + GAMMA * buffer_v[t + 1] - buffer_v[t]
            running_adv = (GAMMA * LAMBDA) * running_adv + delta_t
            advantage.append(running_adv)
          advantage.reverse()
          adv = np.array(advantage)[:, np.newaxis]
          buffer_v.pop()

          bs, ba = np.vstack(buffer_s), np.vstack(buffer_a)
          discounted_r = adv + np.array(buffer_v)[:, np.newaxis]
          buffer_s, buffer_a, buffer_r = [], [], []
          buffer_v = []
          ppo.push_buffer(bs, ba, discounted_r, adv)
        if len(ppo.brain_buffer_s) >= TIME_HORIZON:
          ppo.update()

        if done:
          break;
      print(
        self.name,
        '|Ep: %i' % GLOBAL_EP,
        "|Ep_r: %f" % ep_r
      )
      GLOBAL_EP += 1
      if GLOBAL_EP % MODEL_SAVE_INTERVAL == 0:
        saver.save(sess, MODEL_DIR + "/ppo_model_ep_" + str(GLOBAL_EP) + ".ckpt")

if __name__=="__main__":
  config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True,
    #gpu_options=tf.GPUOptions(
    #  visible_device_list="0",
    #  allow_growth=True
    #)
  )
  sess = tf.Session(config=config)
  #with tf.Session(config=config) as sess:
  with tf.device("/cpu:0"):
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    ppo = PPO(sess)
    workers = []
    for i in range(NUM_WORKERS):
      worker_name="W_%i" % i
      workers.append(Worker(worker_name, ppo))

    summary_ops, summary_vars = build_summaries()
    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    saver = tf.train.Saver()

    nn_model = NN_MODEL
    if nn_model is not None:
      saver.restore(sess, nn_model)

    worker_threads = []
    for worker in workers:
      job = lambda:worker.run()
      t = threading.Thread(target=job)
      t.start()
      worker_threads.append(t)
    with COORD.stop_on_exception():
      COORD.join(worker_threads)

