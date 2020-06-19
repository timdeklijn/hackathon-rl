from gym.envs.registration import register
import gym
import time


class QTable:
    def __init__(self, env):
        self.env = env
        self.max_episodes = 1000

    def run(self):
        for ep in range(self.max_episodes):

            done = False
            state = self.env.reset()
            tot_reward = 0
            while not done:
                action = self.env.action_space.sample()
                new_state, reward, done, _ = self.env.step(action)
                tot_reward += reward
            print(tot_reward)



if __name__ == "__main__":
    register(
        id="FrozenLakeNoSlip-v1",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={"map_name": "4x4", "is_slippery": False},
    )

    env = gym.make("FrozenLakeNoSlip-v1")

    q_table = QTable(env)
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
