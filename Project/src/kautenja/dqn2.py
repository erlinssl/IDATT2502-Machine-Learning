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
    def __init__(self, action_space, in_channels=5):
        super(DQNAgent, self).__init__()
        # print("action", action_space)
        self.dense = nn.Linear(in_channels, 256)
        self.dense2 = nn.Linear(256, action_space)

    def forward(self, x):
        # print("forward", x)
        x = F.relu(self.dense(x))
        x = F.relu(self.dense2(x))
        # print("out shape", x)
        return x

    def save(self):
        if not os.path.exists(os.path.dirname(TRAIN_PATH)):
            os.makedirs(os.path.dirname(TRAIN_PATH))

        torch.save(model.state_dict(), TRAIN_PATH)

    def load(self):
        model = DQNAgent(nn.Module)
        model.load_state_dict(torch.load(TRAIN_PATH))
        model.eval()


env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = DQNAgent(env.action_space.n)
memory = ReplayMemory(64000)
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
    if len(memory) < 512:
        return
    transitions = memory.sample(512)
    batch_current_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_current_state = torch.cat(batch_current_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)

    # print("debug", model(batch_current_state))
    # print("debug", batch_action)

    current_q_values = model(batch_current_state).gather(1, batch_action.type(torch.int64))
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (0.8 * max_next_q_values)  # 0.8 = discount rate

    loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values.squeeze())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


piece_dict = {'T': 0, 'J': 1, 'Z': 2, 'O': 3, 'S': 4, 'L': 5, 'I': 6}
for i_episode in range(50):
    rewards = []
    total_reward = 0
    current_state = env.reset()
    _, reward, done, info = env.step(0)
    next_state = torch.Tensor([[piece_dict[info['current_piece'][0:1]], info['number_of_lines'],
                                info['score'], piece_dict[info['next_piece'][0:1]], info['board_height']]])
    current_state = next_state
    t = 0
    while not done:
        t += 1
        # print("info", next_state)
        env.render()
        action = select_action(next_state)
        if isinstance(action, int):
            action = torch.Tensor([action])
        # print("aaaaaaaaaaaaaaaaaction", action)
        _, reward, done, info = env.step(action.item())
        next_state = torch.Tensor([[piece_dict[info['current_piece'][0:1]], info['number_of_lines'],
                                    info['score'], piece_dict[info['next_piece'][0:1]], info['board_height']]])
        # if done:
        #     reward -= 10

        memory.append((current_state, torch.FloatTensor([[action]]),
                       next_state, torch.FloatTensor([reward])))

        learn()

        current_state = next_state

        total_reward += reward
        rewards.append(total_reward)

        if done:
            print("Episode {epnum: <3} exited after {stepnum: <3} steps".format(
                epnum=i_episode,
                stepnum=t,
            ))
            plot.plot(rewards)
            plot.title("Reward during episode {epnum}"
                       .format(epnum=i_episode))
            plot.show()
            break

model.save()
