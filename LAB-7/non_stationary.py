import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, num_arms=10, total_steps=10000):
        self.num_arms = num_arms
        self.total_steps = total_steps
        self.reward_means = np.zeros(num_arms)
        self.reward_history = []
        
    def update_means(self):
        self.reward_means += np.random.normal(0, 0.01, self.num_arms)
    
    def get_reward(self, action):
        return np.random.normal(self.reward_means[action], 1)
    
    def perform_action(self, action):
        reward = self.get_reward(action)
        self.update_means()
        self.reward_history.append(reward)
        return reward

class EpsilonGreedyAgent:
    def __init__(self, num_arms=10, epsilon=0.1, learning_rate=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q_estimates = np.zeros(num_arms)
        self.action_counts = np.zeros(num_arms)
    
    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            return np.argmax(self.q_estimates)

    def update_estimates(self, action, reward):
        self.q_estimates[action] += self.learning_rate * (reward - self.q_estimates[action])
        self.action_counts[action] += 1

def run_simulation(bandit, agent, total_steps=10000):
    rewards = np.zeros(total_steps)
    actions = np.zeros(total_steps)
    
    for step in range(total_steps):
        action = agent.choose_action()
        reward = bandit.perform_action(action)
        agent.update_estimates(action, reward)
        
        rewards[step] = reward
        actions[step] = action
    
    return rewards, actions

bandit = NonStationaryBandit(num_arms=10, total_steps=10000)
agent = EpsilonGreedyAgent(num_arms=10, epsilon=0.1, learning_rate=0.7)

rewards, actions = run_simulation(bandit, agent, total_steps=10000)

plt.figure(figsize=(10, 5))
cumulative_rewards = np.cumsum(rewards)
average_rewards = cumulative_rewards / (np.arange(1, len(rewards) + 1))
plt.plot(average_rewards, color='blue', label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Epsilon-Greedy Agent Performance in Non-Stationary Environment')
plt.legend()
plt.grid(True)
plt.show()