import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from jordi_modules import processing as wrap
from jordi_modules.dqn import DQN
from jordi_modules.memory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VERSION = "A1_v4_2_"
TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'trained/{}current_best.pt'.format(VERSION))
TENTH_PATH = os.path.join(os.path.dirname(__file__), 'trained/{}every_tenth.pt'.format(VERSION))

GAMMA = 0.9
BATCH_SIZE = 2 ** 6
REPLAY_SIZE = 2 ** 16
LEARN_RATE = 1e-4
SYNC_FRAMES = 1000  # Target network will be synced with main network every n-th frame
REPLAY_START = 2 ** 16
TARGET_REWARD = 9

EPS_START = 1.0
EPS_DECAY = 0.999985
EPS_MIN = 0.02


class Agent:
    def __init__(self, env, mem_buffer):
        self.env = env
        self.mem_buffer = mem_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        with torch.no_grad():
            done_reward = None

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_a = np.array([self.state], copy=False)
                state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())

            new_state, reward, done, info = self.env.step(action)
            self.total_reward += reward * 10  # TODO Stay alive bonus?

            transition = (self.state, action,
                          reward, done, new_state)  # DIFF
            self.mem_buffer.append(transition)

            self.state = new_state

            if done:
                done_reward = self.total_reward - 1
                self._reset()

            return done_reward


def get_loss(batch, net, tgt_net, device=device):
    states, actions, rewards, dones, next_states = zip(*batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    env = gym_tetris.make('TetrisA-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrap.wrap_env(env)

    net = DQN(env.observation_space.shape,
              env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape,
                     env.action_space.n).to(device)

    memory = ReplayMemory(REPLAY_SIZE)
    agent = Agent(env, memory)
    epsilon = EPS_START

    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    total_rewards = []
    step = 0
    last_steps = 0
    best_mean_reward = None

    while True:
        step += 1
        epsilon = max(epsilon * EPS_DECAY, EPS_MIN)

        if len(total_rewards) % 10 == 0:
            env.render()

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print("episode {episodes: <3} : {epstep: <5} / {step: <10}, ended with {reward: <5}, "
                  "mean reward {mean: <8}, (epsilon {eps: <20})"
                  .format(step=step,
                          episodes=len(total_rewards),
                          reward=reward,
                          mean=np.round(mean_reward, 2),
                          eps=epsilon,
                          epstep=step-last_steps))

            last_steps = step

            if best_mean_reward is None or best_mean_reward < mean_reward:
                if not os.path.exists(os.path.dirname(TRAIN_PATH)):
                    os.makedirs(os.path.dirname(TRAIN_PATH))
                torch.save(net.state_dict(), TRAIN_PATH)
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated to {}".format(best_mean_reward))

            if mean_reward > TARGET_REWARD:
                print("Target reached after {} steps".format(step))
                break

            if len(total_rewards) % 10 == 0:
                if not os.path.exists(os.path.dirname(TENTH_PATH)):
                    os.makedirs(os.path.dirname(TENTH_PATH))
                torch.save(net.state_dict(), TENTH_PATH)

            optimizer.zero_grad()
            batch = memory.sample(BATCH_SIZE)
            loss_t = get_loss(batch, net, target_net, device=device)
            loss_t.backward()
            optimizer.step()

        if len(memory) < REPLAY_START:
            continue

        if step % SYNC_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())
