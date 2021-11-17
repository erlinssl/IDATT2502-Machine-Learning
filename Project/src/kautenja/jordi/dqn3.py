import time

import matplotlib.pyplot as plot
from nes_py.wrappers import JoypadSpace
import os
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import random
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'kaut_model.pt')
EPS_DECAY = 2.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(nn.Module):
    def __init__(self, action_space, in_shape):
        super(DQNAgent, self).__init__()
        # print("action", action_space)
        print("in_shape", in_shape)

        self.conv = nn.Sequential(
            nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self.get_conv_out(in_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def get_conv_out(self, shape):
        zeros = torch.zeros(1, *shape)
        print(zeros.shape)
        o = self.conv(zeros)
        return int(np.prod(o.size()))

    def forward(self, x):
        print(x.shape)
        x = self.conv(x).view(x.size()[0], -1)
        x = self.fc(x)
        print(x.shape)
        # print("out shape", x)
        return x

    def save(self):
        if not os.path.exists(os.path.dirname(TRAIN_PATH)):
            os.makedirs(os.path.dirname(TRAIN_PATH))

        torch.save(model.state_dict(), TRAIN_PATH)

    def load(self):
        model = DQNAgent(nn.Module, (240, 256, 3))
        model.load_state_dict(torch.load(TRAIN_PATH))
        model.eval()


env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = DQNAgent(env.action_space.n, env.observation_space.shape)
BATCH_SIZE = 512
memory = ReplayMemory(BATCH_SIZE * 256)
optimizer = optim.RMSprop(model.parameters())
steps_done = 0
episode_durations = []


def exploration_rate(n: int, min_rate=0.01) -> float:
    global t
    return max(min_rate, 1 / (EPS_DECAY ** n + t / 250))


def select_action(state):
    global steps_done
    if np.random.random() > exploration_rate(steps_done):
        with torch.no_grad():
            amp = model(state).type(torch.FloatTensor).data.max(1).indices
            # print("grrr", amp)
            # print(type(amp))
            return amp
    else:
        amp = torch.IntTensor([env.action_space.sample()])
        # print("exploing", amp)
        # print(type(amp))
        return amp


def learn():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch_current_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_current_state = torch.cat(batch_current_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)

    # print("debug", model(batch_current_state))
    # print("debug", batch_action)

    current_q_values = model(batch_current_state).gather(1, batch_action.type(torch.int64))
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (0.95 * max_next_q_values)  # 0.8 = discount rate

    loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values.squeeze())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


piece_dict = {'T': 0, 'J': 1, 'Z': 2, 'O': 3, 'S': 4, 'L': 5, 'I': 6}

ext_pieces = {'Td': 0, 'Tr': 1, 'Tu': 2, 'Tl': 3,
              'Jd': 4, 'Jr': 5, 'Ju': 6, 'Jl': 7,
              'Zh': 8, 'Zv': 9,
              'O': 10,
              'Sh': 11, 'Sv': 12,
              'Ld': 13, 'Lr': 14, 'Lu': 15, 'Ll': 16,
              'Ih': 17, 'Iv': 18, None: 19}
episode_rewards = []
for i_episode in range(50):
    rewards = []
    total_reward = 0
    current_state = env.reset()
    t = 0
    done = False
    while not done:
        t += 1
        # print("info", next_state)
        # env.render()
        action = select_action(torch.FloatTensor([current_state]))
        if isinstance(action, int):
            action = torch.Tensor([action])
        # print("aaaaaaaaaaaaaaaaaction", action)
        next_state, reward, done, info = env.step(action.item())
        if info['current_piece'] is None:
            print(info)
        if info['number_of_lines'] is None:
            print(info)
            print(info['number_of_lines'])

        if done:
            reward -= 10

        # if reward == 0:  # Stay-alive bonus
        #     reward += 1

        if reward != 0:
            print(reward)

        memory.append((current_state, torch.FloatTensor([[action]]),
                       next_state, torch.FloatTensor([reward])))

        learn()

        current_state = next_state

        total_reward += reward
        rewards.append(total_reward)

        if done:
            print("Episode {epnum: <3} exited after {stepnum: <3} steps with a total reward of {reward}".format(
                reward=total_reward,
                epnum=i_episode,
                stepnum=t,
            ))
            episode_rewards.append(total_reward)
            plot.plot(rewards)
            plot.title("Reward during episode {epnum}"
                       .format(epnum=i_episode))
            plot.show()
            break

plot.plot(episode_rewards)
plot.title("Total reward per episode")
plot.show()
model.save()
