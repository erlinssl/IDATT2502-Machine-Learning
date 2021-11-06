import os
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'trained_model.pt')


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
    def __init__(self, action_space, in_channels=4):
        super(DQNAgent, self).__init__()
        self.dense = nn.Linear(in_channels, 256)
        self.dense2 = nn.Linear(256, action_space)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.relu(self.dense2(x))
        return x

    def save(self):
        if not os.path.exists(os.path.dirname(TRAIN_PATH)):
            os.makedirs(os.path.dirname(TRAIN_PATH))

        torch.save(model.state_dict(), TRAIN_PATH)
        print("Model saved!")

    def load(self):
        self.load_state_dict(torch.load(TRAIN_PATH))
        print("Model loaded!")
        model.eval()


env = gym.make('CartPole-v0')

model = DQNAgent(env.action_space.n)
model.load()
memory = ReplayMemory(640)
optimizer = optim.RMSprop(model.parameters())
steps_done = 0
episode_durations = []


def exploration_rate(n: int, min_rate=0.1) -> float:
    return max(min_rate, min(1, 1.0 - np.log10((n + 1) / 25)))


def select_action(state):
    global steps_done
    if np.random.random() < exploration_rate(steps_done):
        with torch.no_grad():
            return model(state).type(torch.FloatTensor).data.max(1)[1].view(1, 1)
    else:
        return env.action_space.sample()


def learn():
    if len(memory) < 32:
        return
    transitions = memory.sample(32)
    batch_current_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_current_state = torch.cat(batch_current_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)

    current_q_values = model(batch_current_state).gather(1, batch_action)
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (0.8 * max_next_q_values)  # 0.8 = discount rate

    loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values.squeeze())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


train = False
if train:
    for i_episode in range(100):
        current_state = env.reset()

        for t in range(200):
            env.render()
            action = select_action(torch.FloatTensor([current_state]))
            next_state, reward, done, _ = env.step(action.item())

            if done and t < 199:
                reward = -1

            memory.append((torch.FloatTensor([current_state]), action,
                           torch.FloatTensor([next_state]), torch.FloatTensor([reward])))

            learn()

            current_state = next_state

            if done:
                print("Episode {epnum: <3} exited after {stepnum: <3} steps".format(
                    epnum=i_episode,
                    stepnum=t,
                ))
                break
else:
    for i_episode in range(10):
        current_state = env.reset()
        for t in range(200):
            env.render()
            action = select_action(torch.FloatTensor([current_state]))
            next_state, reward, done, _ = env.step(action.item())

            if done and t < 199:
                reward = -1

            memory.append((torch.FloatTensor([current_state]), action,
                           torch.FloatTensor([next_state]), torch.FloatTensor([reward])))

            current_state = next_state

            if done:
                print("Episode {epnum: <3} exited after {stepnum: <3} steps".format(
                    epnum=i_episode,
                    stepnum=t,
                ))
                break


# model.save()
