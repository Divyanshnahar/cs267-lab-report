import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, num_arms=10, epsilon=0.1, total_steps=1000):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.total_steps = total_steps
        self.q_estimates = np.zeros(num_arms)
        self.action_counts = np.zeros(num_arms)
        self.reward_means = np.zeros(num_arms)
        self.reward_history = []

    def update_rewards(self):
        self.reward_means += np.random.normal(0, 0.01, self.num_arms)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            return np.argmax(self.q_estimates)

    def perform_step(self):
        action = self.select_action()
        reward = np.random.normal(self.reward_means[action], 1)
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

        self.update_rewards()
        self.reward_history.append(reward)
        return action, reward

    def simulate(self):
        for _ in range(self.total_steps):
            self.perform_step()
        return self.q_estimates, self.reward_history

bandit = NonStationaryBandit(num_arms=10, epsilon=0.1, total_steps=10000)
final_q_values, total_rewards = bandit.simulate()
print("Estimated Q-values:", final_q_values)
print("Total reward collected:", sum(total_rewards))

cumulative_rewards = np.cumsum(total_rewards)
average_rewards = cumulative_rewards / (np.arange(1, len(total_rewards) + 1))
plt.plot(average_rewards, color='red', label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Steps')
plt.grid(True)
plt.legend()
plt.show()