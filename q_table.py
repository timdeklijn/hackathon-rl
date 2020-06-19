from gym.envs.registration import register
import gym
import time
import numpy as np
import matplotlib.pyplot as plt


class QTable:
    def __init__(self, env):
        self.env = env
        self.max_episodes = 10000
        self.epsilon_scale_value = 1000
        self.gamma = 0.9
        self.reward_list = []

        self.size = int(np.sqrt(self.env.observation_space.n))
        self.Q = np.zeros((self.env.observation_space.n, 4))
        self.plot_epsilon()

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
            self.reward_list.append(reward)
        self.plot_cummulative_reward()
        state_list = self.show_policy()
        self.plot_highest_q_values(state_list)

    def plot_highest_q_values(self, state_list):
        max_list = np.array([max(q) for q in self.Q])
        max_list = max_list.reshape((self.size, self.size))
        print(state_list)
        xc, yc = [], []
        for state in state_list:
            xc.append(state % self.size)
            yc.append(state // self.size)
        plt.clf()
        plt.imshow(max_list)
        plt.colorbar()
        plt.scatter(xc, yc, c="r")
        plt.show()

    def show_policy(self):
        state_list = []
        state = self.env.reset()
        done = False
        state_list.append(state)
        while not done:
            action = np.argmax(self.Q[state, :])
            new_state, reward, done, _ = self.env.step(action)
            env.render()
            state = new_state
            state_list.append(new_state)
            time.sleep(0.1)
        return state_list

    def plot_cummulative_reward(self):
        x = [i for i, _ in enumerate(self.reward_list)]
        tot = 0
        y = []
        for r in self.reward_list:
            tot += r
            y.append(tot)
        plt.clf()
        plt.plot(x, y)
        plt.xlabel("episode")
        plt.ylabel("cummulative reward")
        plt.savefig("cummuative_reward.png")

    def change_epsilon(self, ep):
        return 1.0 / ((ep // self.epsilon_scale_value) + 1)

    def plot_epsilon(self):
        x, y = zip(*[[i, self.change_epsilon(i)] for i in range(self.max_episodes)])
        plt.plot(x, y)
        plt.xlabel("episode")
        plt.ylabel("epsilon")
        plt.title("Epsilon / episode")
        plt.savefig("epsilon.png")


if __name__ == "__main__":
    register(
        id="FrozenLakeNoSlip-v1",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={"map_name": "8x8", "is_slippery": False},
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
