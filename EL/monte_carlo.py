from cmath import exp
import math
from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value

class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon = 0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count = 1000, gamma = 0.9, 
              learning_rate = 0.1, render = False, report_interval = 50):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))
        actions = list(range(env.action_space.n))

        for e in range(episode_count):
            s = env.reset()
            done = False
            # 1. play until the end of episode
            experince = []
            while not done:
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                experince.append({"state": s, "action": a, "reward": reward})
                s = n_state
            
            else:
                self.log(reward)

            # 2. evaluate each state, action
            for i, x in enumerate(experince):
                s, a = x["state"], x["action"]

                # calculate discounted future reward of s
                G, t = 0 , 0
                for j in range(i, len(experince)):
                    G += math.pow(gamma, t) * experince[j]["reward"]
                    t += 1
                
                N[s][a] += 1
                alpha = 1 / N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])
            
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode = e)
    
def train():
    agent = MonteCarloAgent()
    env = gym.make("FrozenLakeEasy-v1")
    agent.learn(env, episode_count = 500)
    show_q_value(agent.Q)
    agent.show_reward_log()

if __name__ == "__main__":
    train()