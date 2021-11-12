import math
import time
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
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1.25
BATCH_SIZE = 128
CAPACITY = BATCH_SIZE * 256


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
        self.memory = ReplayMemory(CAPACITY)
        self.dense = nn.Linear(in_channels, 64)
        self.dense2 = nn.Linear(64, 256)
        self.dense3 = nn.Linear(256, 64)
        self.dense4 = nn.Linear(64, action_space)
        self.dense21 = nn.Linear(in_channels, 256)
        self.dense22 = nn.Linear(256, action_space)

    def forward(self, x):
        x = F.relu(self.dense21(x))
        x = F.relu(self.dense22(x))
        print("out", x)
        return x

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch_current_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_current_state = torch.cat(batch_current_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_next_state = torch.cat(batch_next_state)

        # print(batch_current_state.shape)
        print("debug", self(batch_current_state))
        print("batchaction shape", batch_action)
        # print(model(batch_current_state).gather(1, batch_action).shape)
        time.sleep(5)
        current_q_values = self(batch_current_state).gather(1, batch_action)
        max_next_q_values = self(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (0.8 * max_next_q_values)  # 0.8 = discount rate

        loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def save(self):
        if not os.path.exists(os.path.dirname(TRAIN_PATH)):
            os.makedirs(os.path.dirname(TRAIN_PATH))

        torch.save(self.state_dict(), TRAIN_PATH)
        print("Model saved!")

    def load(self):
        self.load_state_dict(torch.load(TRAIN_PATH))
        print("Model loaded!")
        self.eval()


env = gym.make('CartPole-v0')

model = DQNAgent(env.action_space.n)
steps_done = 0
expl_steps = 0
optimizer = optim.RMSprop(model.parameters())
episode_durations = []


def exploration_rate(n: int, min_rate=0.01) -> float:
    return max(min_rate, 1 / EPS_DECAY ** n)
    # max(min_rate, min(1, 1.0 + np.log10(n + 1 / 25)))


def select_action(state):
    global steps_done
    global expl_steps
    rand = round(np.random.random(), 1)
    expl = float(exploration_rate(steps_done))
    # print(steps_done, ": ", rand, "<", expl)
    # time.sleep(0.5)
    if rand > expl:
        # print("woweofjwapjfewp")
        with torch.no_grad():
            print(model(state).type(torch.FloatTensor).data.max(1)[1].view(1, 1))
            # time.sleep(1)
            return model(state).type(torch.FloatTensor).data.max(1)[1].view(1, 1)
    else:
        expl_steps += 1
        # print("eplox:", rand, "<", expl, (rand < expl))
        return env.action_space.sample()


train = True
if train:
    print("Training!")
    for i_episode in range(250):
        current_state = env.reset()
        expl_steps = 0
        for t in range(200):
            # env.render()
            action = select_action(torch.FloatTensor([current_state]))
            # time.sleep(1)
            if isinstance(action, int):
                # print("isint")
                next_state, reward, done, _ = env.step(action)
                action = torch.tensor([[action]])
            else:
                # print("notint")
                next_state, reward, done, _ = env.step(action.item())

            if done and t < 199:
                reward = -1

            # print(action)
            # time.sleep(1)
            model.memory.append((torch.FloatTensor([current_state]), action,
                                 torch.FloatTensor([next_state]), torch.FloatTensor([reward])))

            model.learn()

            current_state = next_state

            if done:
                steps_done += 1
                print("Episode {epnum: <3} exited after {stepnum: <3} steps, {expl: <3} explorational".format(
                    epnum=i_episode,
                    stepnum=t,
                    expl=expl_steps,
                ))
                break
    model.save()
    torch.save(optimizer.state_dict(), OPTIM_PATH)
else:
    print("Testing!")
    model.load()
    model.eval()
    optimizer.load_state_dict(torch.load(OPTIM_PATH))
    for i_episode in range(10):
        current_state = env.reset()
        for t in range(200):
            env.render()
            action = select_action(torch.FloatTensor([current_state]))
            next_state, reward, done, _ = env.step(action)

            if done and t < 199:
                reward = -1

            model.memory.append((torch.FloatTensor([current_state]), action,
                           torch.FloatTensor([next_state]), torch.FloatTensor([reward])))

            if done:
                print("Episode {epnum: <3} exited after {stepnum: <3} steps".format(
                    epnum=i_episode,
                    stepnum=t,
                ))
                break

# print("Model's state_dict:")
# for params in model.state_dict():
#     print(params, '\t', model.state_dict()[params].size())

# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])
