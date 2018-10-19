import tensorflow as tf
import numpy as np
import gym
from ppo import PPO

import myenv

#GAME = 'Pendulum-v0'
GAME = 'myenv-v2'
MAX_EP_STEP = 1000
MAX_EP = 10
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
NN_MODEL = './models/ppo_model_ep_5200.ckpt'
env = gym.make(GAME)

def main():
  #with tf.Session() as sess:
  sess = tf.Session()
  with tf.device("/cpu:0"):
    agent = PPO(sess)
    saver = tf.train.Saver()
    saver.restore(sess, NN_MODEL)

    for ep in range(MAX_EP):
      s = env.reset().reshape(-1)
      ep_r = 0
      for t in range(MAX_EP_STEP):
        env.render()

        #s = np.array([s])
        a = agent.choose_action(s).reshape(-1)

        s_, r, done, info = env.step(a)
        if t ==  MAX_EP_STEP-1:
          done = True
        ep_r += r
        s = s_.reshape(-1)
        if done:
          break
      print(ep, ep_r, done,t)


if __name__ == '__main__':
  main()
