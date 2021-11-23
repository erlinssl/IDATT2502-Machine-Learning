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


piece_dict = {'T': 0, 'J': 1, 'Z': 2, 'O': 3, 'S': 4, 'L': 5, 'I': 6}

print("main")
done = True
for episode in range(50):
    state = env.reset()
    for step in range(5000):
        # action = actionspacetest(step)
        if step % 3 == 0:
            action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        # piece = info['current_piece'][0:1]
        # print(piece, piece_dict[piece])
        env.render()
        # time.sleep(1)
        # print(state)

env.close()
