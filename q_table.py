from gym.envs.registration import register
import gym
import time
import matplotlib.pyplot as plt


class QTable:
    def __init__(self, env):
        self.env = env
        self.max_episodes = 1000
        self.epsilon = 0.99

    def run(self):
        one_count = 0
        for ep in range(self.max_episodes):
            done = False
            state = self.env.reset()
            tot_reward = 0
            while not done:
                action = self.env.action_space.sample()
                new_state, reward, done, _ = self.env.step(action)
                tot_reward += reward
            if tot_reward == 1.0:
                one_count += 1
        print(f"Reached the goal {one_count} times!!")

    def change_epsilon(self, ep):
        # should reduce self.epsilon based on episode number
        return self.epsilon * ep

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
    q_table.plot_epsilon()
    # q_table.run()

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
