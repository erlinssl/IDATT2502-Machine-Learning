from nes_py.wrappers import JoypadSpace
import gym_tetris
import time
from gym_tetris.actions import SIMPLE_MOVEMENT

env = gym_tetris.make('TetrisA-v0')

# print(env.env.__doc__)

env = JoypadSpace(env, SIMPLE_MOVEMENT)


# print(env.observation_space)
# time.sleep(10)


def actionspacetest(step):
    if step % 500 == 0:
        time.sleep(0.5)
        print("incoming")
        return 1
    else:
        return 0


print("main")
done = True
for step in range(5000):
    if done:
        state = env.reset()
    # action = actionspacetest(step)
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()
    print(state)

env.close()
