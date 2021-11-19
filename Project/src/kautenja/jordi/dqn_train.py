import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from jordi_modules import processing as wrap
from jordi_modules.dqn import DQN
from jordi_modules.memory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VERSION = "A1_v5_3"
TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'trained/{}_current_best.pt'.format(VERSION))
TENTH_PATH = os.path.join(os.path.dirname(__file__), 'trained/{}_every_tenth.pt'.format(VERSION))

TRAIN_OPTIM = os.path.join(os.path.dirname(__file__), 'trained/{}_current_best_optim.pt'.format(VERSION))
TENTH_OPTIM = os.path.join(os.path.dirname(__file__), 'trained/{}_every_tenth_optim.pt'.format(VERSION))

GAMMA = 0.9
BATCH_SIZE = 2 ** 6
REPLAY_SIZE = 2 ** 16
LEARN_RATE = 1e-4
SYNC_NTH = 1000  # Target network will be synced with main network every n-th step
REPLAY_START = 2 ** 16
TARGET_REWARD = 500000

EPS_START = 1.0
EPS_DECAY = 0.999985
EPS_MIN = 0.05


class Agent:
    def __init__(self, env, mem_buffer):
        self.env = env
        self.mem_buffer = mem_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0
        self.steps_alive = 0
        self.alive_reward = 0

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
            self.steps_alive += 1
            self.alive_reward += self.steps_alive/25
            self.total_reward += reward * 1000 + (self.steps_alive/25)  # TODO reward for good builds?

            transition = (self.state, action,
                          reward, done, new_state)
            self.mem_buffer.append(transition)

            self.state = new_state

            if done:
                done_reward = self.total_reward - self.alive_reward
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

    loss_t = None
    save_optim = True

    LOAD_PATH = os.path.join(os.path.dirname(__file__), 'trained/30x30/A1_v4_3_again/ep20_oneline_optim.pt'.format(VERSION))
    if save_optim and 0:  # 'and 1' if resuming training from checkpoint
        checkpoint = torch.load(LOAD_PATH)
        print(checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state-dict'])  # ...state_dict
        step = checkpoint['epoch']  # 'step'
        loss_t = checkpoint['loss']
        print("Checkpoint loaded")

    while True:
        step += 1
        epsilon = max(epsilon * EPS_DECAY, EPS_MIN)

        # if len(total_rewards) % 10 == 0:
        env.render()

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.sum(total_rewards[-100:])/100  # so one random early-line doesn't outweight several later
            print("Episode {episodes: <2} : {epstep: <5} / {step: <10}, ended with {reward: <7}, "
                  "mean reward {mean: <9}, (epsilon {eps: <18})"
                  .format(step=step,
                          episodes=len(total_rewards),
                          reward=reward,
                          mean=np.round(mean_reward, 2),
                          eps=epsilon,
                          epstep=step-last_steps))

            last_steps = step

            if best_mean_reward is None or best_mean_reward < mean_reward:
                if save_optim:
                    if not os.path.exists(os.path.dirname(TRAIN_OPTIM)):
                        os.makedirs(os.path.dirname(TRAIN_OPTIM))
                    torch.save({
                        'step': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_t
                    }, TRAIN_OPTIM)
                    print("Saved with optimizer")
                else:
                    if not os.path.exists(os.path.dirname(TRAIN_PATH)):
                        os.makedirs(os.path.dirname(TRAIN_PATH))
                    torch.save(net.state_dict(), TRAIN_PATH)
                    print("state_dict saved")
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated to {}".format(best_mean_reward))

            if mean_reward > TARGET_REWARD:
                print("Target reached after {} steps".format(step))
                break

            if len(total_rewards) % 10 == 0:
                if save_optim:
                    if not os.path.exists(os.path.dirname(TENTH_OPTIM)):
                        os.makedirs(os.path.dirname(TENTH_OPTIM))
                    torch.save({
                        'step': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_t
                    }, TRAIN_OPTIM)
                    print("Tenth saved with optimizer")
                else:
                    if not os.path.exists(os.path.dirname(TENTH_PATH)):
                        os.makedirs(os.path.dirname(TENTH_PATH))
                    torch.save(net.state_dict(), TENTH_PATH)
                    print("Tenth saved")

                plt.title("Total rewards over {} episodes".format(len(total_rewards)))
                plt.plot(np.linspace(0, len(total_rewards)), total_rewards)
                plt.xlabel("Episode")
                plt.ylabel("Reward")

                plt_info = "\n".join((r'$\gamma=%.2f$' % GAMMA,
                                      r'$\varepsilon_start=%.1f$' % EPS_START,
                                      r'$\varepsilon_decay=%.8f$' % EPS_DECAY,
                                      r'$\varepsilon_min  =%.2f$' % EPS_MIN))
                plt.text(0.05, 0.95, plt_info)

                plt.show()

            optimizer.zero_grad()
            batch = memory.sample(BATCH_SIZE)
            loss_t = get_loss(batch, net, target_net, device=device)
            loss_t.backward()
            optimizer.step()

        if len(memory) < REPLAY_START:
            continue

        if step % SYNC_NTH == 0:
            target_net.load_state_dict(net.state_dict())
