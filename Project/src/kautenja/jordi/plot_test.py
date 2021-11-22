import matplotlib.pyplot as plt
import numpy as np

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

total_rewards = [0, 1]

plt.title("Total rewards over {} episodes".format(len(total_rewards)))
plt.plot(np.linspace(0, len(total_rewards), len(total_rewards)), total_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")

plt_info = "\n".join((r'$\gamma=%.2f$' % GAMMA,
                      r'$\varepsilon_start=%.1f$' % EPS_START,
                      r'$\varepsilon_decay=%.8f$' % EPS_DECAY,
                      r'$\varepsilon_min  =%.2f$' % EPS_MIN,
                      r'$batch_size=%d' % BATCH_SIZE,
                      r'$batch_cap=%d' % REPLAY_SIZE))
plt.text(0.0, 0.05, plt_info)

plt.show()
