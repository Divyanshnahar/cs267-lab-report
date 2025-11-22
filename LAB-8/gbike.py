#----------------------------------------------
# Gbike bicycles on rental 
# Value iteration method
# Ref: Reinforcement Learning, Sutton and Barto
# Course: CS308 Artificial Intelligence
# Author: Pratik Shah
# Date: 15 Jan, 2019
# Winter 2019, IIITV
# Gandhinagar
# 
#----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from policy_evaluation_gbike import policy_evaluation_gbike
from policy_improvement_gbike import policy_improvement_gbike

Lambda = np.array([3, 4])  # Rental request arrival
lamda = np.array([3, 2])  # Return

r = 10  # 10 rupee rental reward

t = 2  # 2 rupee transfer fees

policy = np.zeros((21, 21))  # Initial policy of no transfer, transfer policy(i,j) from location 1 to location 2
gam = 0.9

policystable = False
count = 0
while not policystable:
    V = policy_evaluation_gbike(policy, Lambda, lamda, r, t, gam)
    policy, policystable = policy_improvement_gbike(V, policy, Lambda, lamda, r, t, gam)
    count += 1
    # plt.figure(1)
    # plt.subplot(2, 1, 1)
    # plt.contour(policy, [-5, 5])
    # plt.subplot(2, 1, 2)
    # plt.surf(V)
    # plt.pause()

plt.figure(1)
plt.subplot(2, 1, 1)
plt.contour(policy, [-5, 5])
plt.subplot(2, 1, 2)
plt.surf(V)