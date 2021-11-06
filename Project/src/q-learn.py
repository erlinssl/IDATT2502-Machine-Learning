from nes_py.wrappers import JoypadSpace
import gym_tetris
import numpy as np
import time
from gym_tetris.actions import MOVEMENT

env = gym_tetris.make('TetrisA-v0')

print(env.env.__doc__)

env = JoypadSpace(env, MOVEMENT)

Q = np.zeros(env.action_space.n,)


done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    time.sleep(0.05)

env.close()
