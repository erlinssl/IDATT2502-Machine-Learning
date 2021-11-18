import gym
import cv2
import numpy as np
from collections import deque
import random

HEIGHT = 30
WIDTH = 30

class MaxAndSkipEnv(gym.Wrapper):
    '''
    Used to repeat a given action n times, given by the skip parameter. This is done since
    the environment takes a few frames to update anyways, so we can offload some extra work.
    The states of each "skipped" frame that is max pooled then passed on.
    '''
    def __init__(self, env=None, skip=2):
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


class ProcessFrame84(gym.ObservationWrapper):
    '''
    Preprocessing to downscale a gym obersvation from it's original resolution RGB image to
    a grayscaled 20x20 image, which will be a lot easier to pass through the NN.
    Also crops out uncessessary noise, like the sidebars in our environment.
    '''
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]) \
                .astype(np.float32)

        elif frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]) \
                .astype(np.float32)

        else:
            assert False, "Unknown resolution."

        # 95 + 80 / 47 + 160
        img = img[47:208, 95:174, 0] * 0.299 + img[47:208, 95:174, 1] * 0.587 + img[47:208, 95:174, 2] * 0.114
        # cv2.cv2.imshow("before_resize", img)  # For debugging image crop
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        resized_screen = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # cv2.imshow("after_resize", resized_screen)  # For debugging image rescaling
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x_t = np.reshape(resized_screen, [WIDTH, HEIGHT, 1])
        # cv2.imshow("after_reshape", x_t)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x_t = x_t.astype(np.uint8)
        # if random.random() < 0.05:
        #     cv2.imshow("random", x_t)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        return x_t


class BufferWrapper(gym.ObservationWrapper):
    '''
    Giving the agent a single frame of the game won't really tell it anything, so this is used
    to create a buffer with subsequent frames to give the agent an idea of where things are moving
    '''
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
    '''
    Simply used to rearrange a tensors dimensions, since the nn's
    convolution layers expect the color dimension first (C, H, W),
    whereas the gym observations are in the shape (H, W, C)
    '''
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    '''
    Converts tensors with values between 0 and 255 to ones with
    floats in the range [0.0, ..., 1.0], since this is a better
    representation for an NN.
    '''
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def wrap_env(env):
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
