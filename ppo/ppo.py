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

#GAME='Pendulum-v0'
GAME='myenv-v2'
env = gym.make(GAME)
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]
A_BOUNDS = [env.action_space.low, env.action_space.high]
NONE_STATE = np.zeros(NUM_STATES)

EP_MAX = 10000
EP_LEN = 200
GAMMA = 0.9
BATCH = 64
EPOCH = 3
CLIP_EPSILON = 0.2
NUM_HIDDENS = [512, 512, 512]
LEARNING_RATE = 1e-4
BETA = 1e-4# entropy

# directories
LOG_DIR = "./log"
MODEL_DIR = "./models"

MODEL_SAVE_INTERVAL = 100
RENDER_EP = 100

GLOBAL_EP = 6000
NN_MODEL = None
NN_MODEL = "/home/amsl/reinforcement_learning/ppo/models/ppo_model_ep_" + str(GLOBAL_EP) + ".ckpt"
NUM_WORKERS = 32

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
    self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES), name='state')

    self.global_step = tf.Variable(0., trainable=False)
    self.learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step, 1000, 0.98, staircase=True)

    self.weight_init = tf.random_normal_initializer(0.0, 0.1)
    # common
    pi, pi_params = self._build_net('pi', trainable=True)
    oldpi, oldpi_params = self._build_net('oldpi', trainable=False)

    # critic
    with tf.variable_scope('critic'):
      lc = tf.layers.dense(self.s_t, NUM_HIDDENS[0], tf.nn.relu, name="lc")
      self.v = tf.layers.dense(lc, 1, kernel_initializer=self.weight_init, name="value")
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
      la0 = tf.layers.dense(self.s_t, NUM_HIDDENS[0], tf.nn.relu, kernel_initializer=self.weight_init, name="la0")
      la1 = tf.layers.dense(la0, NUM_HIDDENS[1], tf.nn.relu, kernel_initializer=self.weight_init, name="la1")
      mu = tf.layers.dense(la1, NUM_ACTIONS, tf.nn.tanh, kernel_initializer=self.weight_init, name="mu", trainable=trainable)
      sigma = tf.layers.dense(la1, NUM_ACTIONS, tf.nn.softplus, kernel_initializer=self.weight_init, name="sigma", trainable=trainable)
      norm_dist = tf.distributions.Normal(loc=mu * A_BOUNDS[1], scale=sigma)
      #prob = tf.layers.dense(la1, NUM_ACTIONS, tf.nn.softmax, kernel_initializer=self.weight_init, name="prob", trainable=trainable)
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return norm_dist, params

  def update(self, s, a, r):
    self.sess.run(self.update_oldpi_op)

    adv = self.sess.run(self.advantage, {self.s_t: s, self.tfdc_r: r})
    # adv = (adv - adv.mean())/(adv.std()+1e-6)   # sometimes helpful

    for _ in range(EPOCH):
      # update actor
      self.sess.run(self.a_train_op, {self.s_t: s, self.a_t: a, self.adv: adv})

    for _ in range(EPOCH):
      # update critic
      self.sess.run(self.c_train_op, {self.s_t: s, self.tfdc_r: r})

    _feed_dict={self.s_t:s, self.a_t:a, self.tfdc_r:r, self.adv:adv}
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

  def choose_action(self, s):
    s = s[np.newaxis, :]
    a = self.sess.run([self.sample_op], {self.s_t: s})
    return np.clip(a, A_BOUNDS[0], A_BOUNDS[1]).reshape(-1)

  def get_v(self, s):
    if s.ndim < 2: s = s[np.newaxis, :]
    return self.sess.run(self.v, {self.s_t: s})[0, 0]

class Worker:
  def __init__(self, name, brain):
    self.env = gym.make(GAME)
    self.name = name

  def run(self):
    global GLOBAL_EP
    while not COORD.should_stop():
      s = self.env.reset()
      buffer_s, buffer_a, buffer_r = [], [], []
      ep_r = 0
      for t in range(EP_LEN):    # in one episode
        if(self.name=="W_0"):
          self.env.render()
        a = ppo.choose_action(s)
        start_time = time.time()
        s_, r, done, _ = self.env.step(a)
        elapsed_time = time.time() - start_time
        #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        buffer_s.append(s)
        buffer_a.append(a)
        #buffer_r.append((r+8)/8)  # normalize reward, find to be useful
        buffer_r.append(r)
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1 or done:
          v_s_ = ppo.get_v(s_)
          discounted_r = []
          for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
          discounted_r.reverse()

          bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
          buffer_s, buffer_a, buffer_r = [], [], []
          ppo.update(bs, ba, br)

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

