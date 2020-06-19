from gym.envs.registration import register
import gym
import time
import numpy as np
import matplotlib.pyplot as plt


class QTable:
    def __init__(self, env):
        self.env = env
        self.max_episodes = 10000
        self.gamma = 0.9

        self.Q = np.zeros((16, 4))

    def run(self):
        for ep in range(self.max_episodes):
            tot_reward = 0
            epsilon = self.change_epsilon(ep)
            done = False
            state = self.env.reset()
            while not done:

                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state, :])

                new_state, reward, done, _ = self.env.step(action)

                self.Q[state, action] = reward + (
                    self.gamma * np.max(self.Q[new_state])
                )

                tot_reward += reward
                state = new_state
            print(f"{ep}: {reward}")
        self.show_policy()

    def show_policy(self):
        state = self.env.reset()
        done = False
        while not done:
            action = np.argmax(self.Q[state, :])
            new_state, reward, done, _ = self.env.step(action)
            env.render()
            state = new_state
            time.sleep(0.1)

    def change_epsilon(self, ep):
        return 1.0 / ((ep // 1000) + 1)

    def plot_epsilon(self):
        x, y = zip(*[[i, self.change_epsilon(i)] for i in range(self.max_episodes)])
        plt.plot(x, y)
        plt.show()


if __name__ == "__main__":
    register(
        id="FrozenLakeNoSlip-v1",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={"map_name": "4x4", "is_slippery": False},
    )

    env = gym.make("FrozenLakeNoSlip-v1")

    q_table = QTable(env)
    # q_table.plot_epsilon()
    q_table.run()

    # print(state)

    # LEFT = 0
    # DOWN = 1
    # RIGHT = 2
    # UP = 3

    # actions = [2, 2, 1, 1, 1, 2]
    # state = env.reset()
    # for action in actions:
    #     new_state, reward, done, _ = env.step(action)
    #     print("===", reward)
    #     env.render()
    #     time.sleep(0.2)
