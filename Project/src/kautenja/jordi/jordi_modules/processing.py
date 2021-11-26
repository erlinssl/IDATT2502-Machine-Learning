import gym
import cv2
import numpy as np
from collections import deque
import random


class MaxAndSkipEnv(gym.Wrapper):
    '''
    Used to repeat a given action n times, given by the skip parameter. This is done since
    the environment takes a few frames to update anyways, so we can offload some extra work.
    The states of each "skipped" frame that is max pooled then passed on.
    '''

    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._state_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            next_state, reward, done, info = self.env.step(action)
            self._state_buffer.append(next_state)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._state_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._state_buffer.clear()
        state = self.env.reset()
        self._state_buffer.append(state)
        return state


HEIGHT = 20
WIDTH = 10


class ProcessFrameXY(gym.ObservationWrapper):
    """
    Preprocessing to downscale a gym obersvation from it's original resolution RGB image to
    a grayscaled 20x10 image, which will be a lot easier to pass through the NN.
    Also crops out uncessessary noise, like the sidebars in our chosen environment.
    """

    def __init__(self, env=None):
        super(ProcessFrameXY, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrameXY.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]) \
                .astype(np.float32)  # general gym.Atari resolution

        elif frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]) \
                .astype(np.float32)  # our tetris env resolution

        else:
            assert False, "Unknown resolution."

        # 95 + 81 / 47 + 161
        img = img[47:208, 95:176, 0] * 0.299 + img[47:208, 95:176, 1] * 0.587 + img[47:208, 95:176, 2] * 0.114
        # cv2.imshow("before_resize", img)  # For debugging image crop
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        resized_screen = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # cv2.imshow("after_resize", resized_screen)  # For debugging image rescaling
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x_t = np.reshape(resized_screen, [HEIGHT, WIDTH, 1])
        # cv2.imshow("after_reshape", x_t)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x_t = x_t.astype(np.uint8)
        # if test > 1500:
        #     print("scale show")
        #     cv2.imshow("random", x_t)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        return x_t


class BufferWrapper(gym.ObservationWrapper):
    """
    Giving the agent a single frame of the game won't really tell it anything, so this is used
    to create a buffer with subsequent frames to give the agent an idea of where things are moving
    """

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0),
                                                dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Simply used to rearrange a tensors dimensions, since the nn's
    convolution layers expect the color dimension first (C, H, W),
    whereas the gym observations are in the shape (H, W, C)
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Converts tensors with values between 0 and 255 to either
    0.0 or 1.0, since this is a better representation for the NN.
    Divisor 125 is somewhat arbitray and may differ depending on
    the environment. For the one I'm using, this ensures all tetromino
    gradients are properly rounded to 1.
    """

    def observation(self, obs):
        obs = np.array(obs).astype(np.float32) / 125
        return np.round(obs)


def wrap_env(env, buffersize: int = 2, skip: int = 4, heuristic=False):
    if skip > 0:
        env = MaxAndSkipEnv(env, skip=skip)
    env = ProcessFrameXY(env)
    env = ImageToPyTorch(env)
    if buffersize > 1:
        env = BufferWrapper(env, buffersize)
    if not heuristic:
        env = ScaledFloatFrame(env)
    return env
