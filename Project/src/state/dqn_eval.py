import cv2
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from modules import processing as wrap
from modules.dqn import DQN
from modules.memory import ReplayMemory
import numpy as np
import os
import torch
import collections

filename = 'ep612_mean10'
DICT_PATH = os.path.join(os.path.dirname(__file__), 'trained/30x30/A1_v4_3_again/{}.pt'.format(filename))

env = gym_tetris.make('TetrisA-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = wrap.wrap_env(env)

net = DQN(env.observation_space.shape, env.action_space.n)
checkpoint = torch.load(DICT_PATH)  # ['model_state_dict']
net.load_state_dict(checkpoint)
net.eval()

rewards = []
for i_episode in range(10):
    total_reward = 0.0
    state = env.reset()
    done = False
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    while not done:
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v)
        action = np.argmax(q_vals.detach().numpy())

        actions[int(action)] += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward

    print("Episode {epnum: <2} ended with {rew: <10}, action record is [{act}]".format(epnum=i_episode,
                                                                                       rew=total_reward,
                                                                                       act=actions))
