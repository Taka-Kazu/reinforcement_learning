# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import gym, time, threading
import numpy as np
import random as pyrandom
from gym import wrappers  # gymの画像保存
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない

ENV = 'Pendulum-v0'
env = gym.make(ENV)
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]
A_BOUNDS = [env.action_space.low, env.action_space.high]
NONE_STATE = np.zeros(NUM_STATES)

# LEARNING PARAMS
LEARNING_RATE = 1e-4
DECAY_STEPS = 1000
DECAY_RATE = 0.98
EPSILON = 0.2
MIN_BATCH = 512
BUFFER_SIZE = MIN_BATCH * 5
MAX_STEPS = 200
EPOCH = 3# 学習回数
LOSS_V = 0.2
LOSS_ENTROPY = 1e-3# beta
LAMBDA = 0.95
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** (N_STEP_RETURN)

# LAYER PARAMS
NUM_HIDDENS = [256, 256, 256]
CELL_SIZE = 64

# DIRECTORIES
MODEL_DIR = './models'
#LOG_DIR = './log'
SUMMARY_DIR = './results'

#TIME_HORIZON = 32

# ε-greedyのパラメータ
EPS_START = 1
EPS_END = 0.01
EPS_STEPS = 500*MAX_STEPS

TARGET_SCORE = 1000#-250
GLOBAL_EP = 0
MODEL_SAVE_INTERVAL = 100
NN_MODEL = None
#NN_MODEL = '/home/amsl/Desktop/ex/ppo_model_ep_44800.ckpt'

def build_summaries():
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
  summary_ops = tf.summary.merge_all()

  return summary_ops, summary_vars

class PPONet(object):
  def __init__(self, sess):
    self.sess = sess
    self.global_step = tf.Variable(0, trainable=False)# 記録用
    self.learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step, DECAY_STEPS, DECAY_RATE, staircase=True)

    self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES), name="s_t")
    self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name="a_t")
    self.r_t = tf.placeholder(tf.float32, shape=(None, 1), name="r_t")

    # alpha, betaは分布を表すパラメータ
    with tf.name_scope("pi"):
      #self.alpha, self.beta, self.v, self.params = self._build_net('pi', trainable=True)
      self.alpha, self.beta, self.params = self._build_net('pi', trainable=True)

    with tf.name_scope("old_pi"):
      #self.old_alpha, self.old_beta, self.old_v, self.old_params = self._build_net('old_pi', trainable=False)
      self.old_alpha, self.old_beta, self.old_params = self._build_net('old_pi', trainable=False)

    with tf.name_scope("critic"):
      weight_init = tf.random_normal_initializer(0., .1)
      step_size = tf.shape(self.s_t)[:1]
      s = tf.expand_dims(self.s_t, axis=1, name="timely_input")
      rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=CELL_SIZE, name="LSTM")
      self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
      outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                     inputs=s,
                                     initial_state=self.init_state,
                                     sequence_length=step_size,
                                     time_major=True
                                     )

      cell_out = tf.reshape(outputs, [-1, CELL_SIZE], name="flatten_outputs")
      l4 = tf.layers.dense(cell_out, NUM_HIDDENS[2], tf.nn.relu, trainable=True, name="l4")
      self.v = tf.layers.dense(l4, 1, trainable=True, name="value")

    self.train_queue = [[], [], [], [], []]# s, a, r, s', s' terminal mask
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    # paramのコピー操作
    self.assign_op = [self.old_params[i].assign(self.params[i]) for i in range(len(self.params))]
    self.graph = self._build_graph()

  def _build_net(self, name, trainable):
    with tf.variable_scope(name):
      l1 = tf.layers.dense(self.s_t, NUM_HIDDENS[0], tf.nn.relu, trainable=trainable, name="l1")
      #l2 = tf.layers.dense(l1, NUM_HIDDENS[1], tf.nn.relu, trainable=trainable, name="l2")
      #l3 = tf.layers.dense(l2, NUM_HIDDENS[2], tf.nn.relu, trainable=trainable, name="l3")
      alpha = tf.layers.dense(l1, NUM_ACTIONS, tf.nn.softplus, trainable=trainable, name="actor_alpha")
      beta = tf.layers.dense(l1, NUM_ACTIONS, tf.nn.softplus, trainable=trainable, name="actor_beta")
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    #return alpha, beta, value, params
    return alpha, beta, params

  def _build_graph(self):
    with tf.name_scope("action"):
      x = (self.a_t - A_BOUNDS[0]) / (-A_BOUNDS[0] + A_BOUNDS[1])

      beta_dist = tf.distributions.Beta(self.alpha + 1, self.beta + 1)
      self.prob = beta_dist.prob(x) + 1e-8

      beta_dist_old = tf.contrib.distributions.Beta(self.old_alpha + 1, self.old_beta + 1)
      self.prob_old = tf.stop_gradient(beta_dist_old.prob(x) + 1e-8)

      self.A = beta_dist.sample(1) * (-A_BOUNDS[0] + A_BOUNDS[1]) + A_BOUNDS[0]

    with tf.name_scope("actor_loss"):
      # 誤差関数
      self.advantage = self.r_t - self.v
      r_theta = self.prob / self.prob_old
      # CPI:Conservative Policy Iteration
      loss_CPI = r_theta * tf.stop_gradient(self.advantage)
      r_clip = tf.clip_by_value(r_theta, 1 - EPSILON, 1 + EPSILON)
      clipped_loss_CPI = r_clip * tf.stop_gradient(self.advantage)
      self.loss_CLIP = -tf.reduce_mean(tf.minimum(loss_CPI, clipped_loss_CPI))

      self.loss_value = LOSS_V * tf.reduce_mean(tf.square(self.advantage))
      self.entropy = LOSS_ENTROPY * tf.reduce_mean(beta_dist.entropy())
      self.loss_total = self.loss_CLIP + self.loss_value - self.entropy

      minimize = self.optimizer.minimize(self.loss_total, global_step=self.global_step)
      return minimize

  def update_parameter_server(self):
    if len(self.train_queue[0]) < BUFFER_SIZE:
        return
    queue = self.train_queue
    self.train_queue = [[], [], [], [], []]
    Buffer = np.array(queue).T
    [self.sess.run(self.assign_op[i]) for i in range(len(self.assign_op))]
    for i in range(EPOCH):
      print("EPOCH:" + str(i+1))
      n_batches = int(BUFFER_SIZE / MIN_BATCH)
      #batch = np.random.permutation(Buffer)
      batch = Buffer
      for n in range(n_batches):
        s, a, r, s_, s_mask = np.array(batch[n * MIN_BATCH: (n + 1) * MIN_BATCH]).T
        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        v = self.get_v(s)

        r = r + LAMBDA * GAMMA_N * v * s_mask

        feed_dict = {self.s_t: s, self.a_t: a, self.r_t: r}

        minimize = self.graph
        self.sess.run(minimize, feed_dict)

    summary_str = self.sess.run(summary_ops, feed_dict = {
      summary_vars[0]: r.mean(),
      summary_vars[1]: self.sess.run(self.entropy, feed_dict={self.s_t:s}) / LOSS_ENTROPY,
      summary_vars[2]: self.sess.run(self.learning_rate),
      summary_vars[3]: self.sess.run(self.loss_CLIP, feed_dict),
      summary_vars[4]: self.sess.run(self.loss_value, feed_dict={self.s_t: s, self.r_t: r}),
      summary_vars[5]: self.sess.run(self.v, feed_dict={self.s_t: s}).mean()
    })
    writer.add_summary(summary_str,GLOBAL_EP)
    writer.flush()

  def predict_a(self, s):    # 状態sから各action
    a = self.sess.run(self.A, feed_dict={self.s_t: s})
    return a

  def train_push(self, s, a, r, s_):
    self.train_queue[0].append(s)
    self.train_queue[1].append(a)
    self.train_queue[2].append(r)

    if s_ is None:
      self.train_queue[3].append(NONE_STATE)
      self.train_queue[4].append(0.)
    else:
      self.train_queue[3].append(s_)
      self.train_queue[4].append(1.)

  def get_v(self, s):
    v = self.sess.run(self.v, feed_dict={self.s_t: s})
    return v

  def reset_lstm(self):
    self.sess.run(self.init_state)

class Agent:
  def __init__(self, brain):
    self.brain = brain
    self.memory = []    # s,a,r,s_の保存メモリ、used for n_step return
    self.R = 0          # 時間割引した、「Nステップ分あとまで」の総報酬R
    self.count = 0

  def act(self, s):
    if frames >= EPS_STEPS:   # ε-greedy法で行動を決定
      eps = EPS_END
    else:
      eps = EPS_START + frames * (EPS_END - EPS_START) / EPS_STEPS  # linearly interpolate

    if np.random.rand() < eps:
      return np.random.uniform(A_BOUNDS[0], A_BOUNDS[1], NUM_ACTIONS)  # ランダムに行動
    else:
      s = np.array([s])
      a = self.brain.predict_a(s).reshape(-1)
      return a

  def advantage_push_brain(self, s, a, r, s_):   # advantageを考慮したs,a,r,s_をbrainに与える
    def get_sample(memory, n):  # advantageを考慮し、メモリからnステップ後の状態とnステップ後までのRを取得する関数
      s, a, _, _ = memory[0]
      _, _, _, s_ = memory[n - 1]
      return s, a, self.R, s_

    self.memory.append((s, a, r, s_))
    # 前ステップの「時間割引Nステップ分の総報酬R」を使用して、現ステップのRを計算
    self.R = (self.R + LAMBDA * r * GAMMA_N) / GAMMA     # r0はあとで引き算している、この式はヤロミルさんのサイトを参照

    # advantageを考慮しながら、LocalBrainに経験を入力する
    if s_ is None:
      while len(self.memory) > 0:
        n = len(self.memory)
        self.R = self.R - LAMBDA * self.memory[0][2] + self.memory[0][2]
        s, a, r, s_ = get_sample(self.memory, n)
        self.brain.train_push(s, a, r, s_)
        self.R = (self.R - self.memory[0][2]) / GAMMA
        self.memory.pop(0)

      self.R = 0  # 次の試行に向けて0にしておく

    if len(self.memory) >= N_STEP_RETURN:
      self.R = self.R - LAMBDA * self.memory[0][2] + self.memory[0][2]
      s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
      self.brain.train_push(s, a, r, s_)
      self.R = self.R - self.memory[0][2]     # # r0を引き算
      self.memory.pop(0)
      #print([self.memory[i][2] for i in range(len(self.memory))])

class Environment:
  total_reward_vec = np.array([0 for i in range(20)])  # 総報酬を20試行分格納して、平均総報酬をもとめる

  def __init__(self, name, brain):
    self.name = name
    self.env = gym.make(ENV)
    self.agent = Agent(brain)    # 環境内で行動するagentを生成
    self.memory = []
    self.thread_step = 0

  def run(self):
    global frames  # セッション全体での試行数
    global isLearned
    global GLOBAL_EP

    s = self.env.reset().reshape(-1)
    R = 0
    step = 0
    self.agent.brain.reset_lstm()
    while True:
      if self.name == 'W_0':
          self.env.render()
      a = self.agent.act(s)
      s_, r, done, info = self.env.step(a)
      s_ = s_.reshape(-1)
      a = a.reshape(-1)
      #r = r.reshape(-1)

      step += 1
      frames += 1     # セッショントータルの行動回数をひとつ増やします

      if step > MAX_STEPS or done:  # terminal state
          s_ = None

      # 報酬と経験を、Brainにプッシュ
      r=(r+8)/8
      self.agent.advantage_push_brain(s, a, r, s_)
      s = s_
      R += r
      if len(self.agent.brain.train_queue[0]) >= BUFFER_SIZE:
          if not isLearned:
              self.agent.brain.update_parameter_server()
      if step > MAX_STEPS or done:
          self.total_reward_vec = np.hstack((self.total_reward_vec[1:], R))
          break

    print(
      self.name,
      "| EP: %d" % GLOBAL_EP,
      "| reward: %f" % R,
      "| step: %d" % step,
      "| running_reward: %f" % self.total_reward_vec.mean(),
      )
    GLOBAL_EP += 1
    if GLOBAL_EP % MODEL_SAVE_INTERVAL == 0:
      saver.save(SESS,MODEL_DIR + "/ppo_model_ep_" + str(GLOBAL_EP) +".ckpt")
    # スレッドで平均報酬が一定を越えたら終了
    #if self.total_reward_vec.mean() > TARGET_SCORE:
    if GLOBAL_EP > 60000:
      isLearned = True
      time.sleep(2.0)     # この間に他のlearningスレッドが止まります

# --スレッドになるクラスです-------
class Worker_thread:
    # スレッドは学習環境environmentを持ちます
    def __init__(self, thread_name, brain):
      self.environment = Environment(thread_name, brain)

    def run(self):
      while true:
        self.environment.run()
        if islearned:
            break

# -- main -----------------------------
if __name__ == "__main__":
  frames = GLOBAL_EP * MAX_STEPS
  isLearned = False

  with tf.Session() as SESS:
    brain = PPONet(SESS)

    summary_ops, summary_vars = build_summaries()
    SESS.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR,SESS.graph)
    saver = tf.train.Saver()

    nn_model = NN_MODEL
    if nn_model is not None:
      saver.restore(SESS,nn_model)
      print("Model restored!!")

    environment = Environment("W_0", brain)
    while True:
      environment.run()
      if isLearned:
          break
