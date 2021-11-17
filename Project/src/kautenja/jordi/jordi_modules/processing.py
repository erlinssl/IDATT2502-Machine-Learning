import cv2.cv2
import gym
import cv2
import torch
import torch.nn as nn
import numpy as np


class ProcessFrame84(gym.ObservationWrapper):
    '''
    Preprocessing to downscale a gym obersvation from it's RGB original resolution to
    a grayscaled 84x84 image, which will be a lot easier to pass through the NN.
    '''
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

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

        # 95 + 81 / 47 + 162
        img = img[47:209, 95:176, 0] * 0.299 + img[47:209, 95:176, 1] * 0.587 + img[47:209, 95:176, 2] * 0.114
        # cv2.cv2.imshow("before_resize", img)  # For debugging image crop
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        # cv2.imshow("after_resize", resized_screen)  # For debugging image rescaling
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


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
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
