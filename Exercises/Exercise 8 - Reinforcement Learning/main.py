# print(env.env.__doc__)
# Description:
#     A pole is attached by an un-actuated joint to a cart, which moves along
#     a frictionless track. The pendulum starts upright, and the goal is to
#     prevent it from falling over by increasing and reducing the cart's
#     velocity.
#
# Source:
#     This environment corresponds to the version of the cart-pole problem
#     described by Barto, Sutton, and Anderson
#
# Observation:
#     Type: Box(4)
#     Num     Observation               Min                     Max
#     0       Cart Position             -4.8                    4.8
#     1       Cart Velocity             -Inf                    Inf
#     2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
#     3       Pole Angular Velocity     -Inf                    Inf
#
# Actions:
#     Type: Discrete(2)
#     Num   Action
#     0     Push cart to the left
#     1     Push cart to the right
#
#     Note: The amount the velocity that is reduced or increased is not
#     fixed; it depends on the angle the pole is pointing. This is because
#     the center of gravity of the pole increases the amount of energy needed
#     to move the cart underneath it
#
# Reward:
#     Reward is 1 for every step taken, including the termination step
#
# Starting State:
#     All observations are assigned a uniform random value in [-0.05..0.05]
#
# Episode Termination:
#     Pole Angle is more than 12 degrees.
#     Cart Position is more than 2.4 (center of the cart reaches the edge of
#     the display).
#     Episode length is greater than 200.
#     Solved Requirements:
#     Considered solved when the average return is greater than or equal to
#     195.0 over 100 consecutive trials.
import math
import time
from typing import Tuple

import gym
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

env = gym.make('CartPole-v0')

# pole angle, angular velocity
n_bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(24)]
upper_bounds = [env.observation_space.high[2], math.radians(24)]

# cart position, cart velocity, pole angle, pole velocity
# n_bins = (12, 6, 6, 12)
# lower_bounds = [env.observation_space.low[0], env.observation_space.low[0]/2,
#                 env.observation_space.low[2], -math.radians(24)]
# upper_bounds = [env.observation_space.high[0], env.observation_space.high[0]/2,
#                 env.observation_space.high[2], math.radians(24)]

Q = np.zeros(n_bins + (env.action_space.n,))
print(lower_bounds)
print(upper_bounds)
print(Q.shape)


def discretizer(cart_pos, cart_vel, angle, pole_velocity) -> Tuple[int, ...]:
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))
    # return tuple(map(int, est.transform([[cart_pos, cart_vel, angle, pole_velocity]])[0]))


def next_action(state: tuple):
    return np.argmax(Q[state])


def update_q(reward: float, new_state: tuple, discount_factor=1) -> float:
    future_optimal_value = np.max(Q[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


def learning_rate(n, min_rate=0.01):
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


def exploration_rate(n: int, min_rate=0.1) -> float:
    return max(min_rate, min(1, 1.0 - np.log10((n + 1) / 25)))


time.sleep(1)
for i_episode in range(1000):
    current_state, done = discretizer(*env.reset()), False

    ex = 0
    for t in range(200):
        env.render()

        action = next_action(current_state)
        obs, rew, done, info = env.step(action)
        new_state = discretizer(*obs)

        if np.random.random() < exploration_rate(i_episode):
            action = env.action_space.sample()
            ex += 1

        lr = learning_rate(i_episode)
        q_new = update_q(rew, new_state)
        q_old = Q[current_state][action]
        # https://en.wikipedia.org/wiki/Q-learning#Algorithm
        Q[current_state][action] = (1 - lr) * q_old + lr * q_new

        current_state = new_state

        if i_episode % 50 == 0:
            time.sleep(0.001)
        if done:
            if t == 199:
                print("Episode {epinum: <3} completed the exercise with {exposteps: <3} explorational steps"
                      .format(epinum=i_episode, exposteps=ex))
                time.sleep(0.01)
                break
            print("Episode {epinum: <3} failed after {tsteps: <3} timesteps, with {exposteps: <3} explorational steps"
                  .format(epinum=i_episode, tsteps=(t + 1), exposteps=ex))
            break
env.close()
