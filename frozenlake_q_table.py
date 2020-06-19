import gym
import numpy as np
from gym.envs.registration import register
import matplotlib.pyplot as plt
import time


class QTableLearner:
    def __init__(self, env):
        self.env = env

        self.gamma = 0.99
        self.num_episodes = 10000
        self.greedy_scale = 1000

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.ep_list = []

        self.plot_epsilon()

    def run(self):
        # Run all episodes
        for ep in range(self.num_episodes):
            # Setup episode
            e_greedy = self._set_e_greedy(ep)
            state = self.env.reset()
            ep_reward, steps, done = 0, 0, False
            # Start episode
            while not done:
                # Choose action
                action = self._choose_action(e_greedy, state)
                # Perform action and get back response
                new_state, reward, done, _ = self.env.step(action)
                # 'learn' from experience
                self.Q[state, action] = reward + self.gamma * np.max(
                    self.Q[new_state, :]
                )
                # Update some episode variables
                ep_reward += reward
                steps += 1
                state = new_state
            # Print episode stats
            print(f"{ep:5d} - Reward: {ep_reward}, Steps: {steps}")

        self.run_and_render()
        self.plot_q_table()

    def _choose_action(self, e_greedy, state):
        if np.random.rand() < e_greedy:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state, :])

    def _set_e_greedy(self, ep):
        return 1.0 / ((ep // self.greedy_scale) + 1)

    def run_and_render(self):
        print("=" * 50 + "\n")
        state = self.env.reset()
        done = False
        while not done:
            action = np.argmax(self.Q[state, :])
            new_state, reward, done, _ = self.env.step(action)
            env.render()
            state = new_state
            time.sleep(0.1)
            print("\n" + "-" * 30 + "\n")

    def plot_epsilon(self):
        x, y = zip(*[[i, self._set_e_greedy(i)] for i in range(self.num_episodes)])
        plt.clf()
        plt.plot(x, y)
        plt.title("Espilon Greedy")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.savefig("epsilon.png")

    def plot_q_table(self):
        side = int(np.sqrt(self.Q.shape[0]))
        action_list = np.array([np.argmax(state) for state in self.Q])
        action_list = action_list.reshape((side, side))
        plt.clf()
        plt.imshow(action_list)
        plt.colorbar()
        plt.title("0:L, 1:D, 2:R, 3:U")
        plt.savefig("q_table.png")


if __name__ == "__main__":
    register(
        id="FrozenLakeNoSlip-v1",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={"map_name": "8x8", "is_slippery": False},
    )

    env = gym.make("FrozenLakeNoSlip-v1")
    q_table_learner = QTableLearner(env)
    q_table_learner.run()
