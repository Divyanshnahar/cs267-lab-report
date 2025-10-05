import random
import math
import matplotlib.pyplot as plt
import copy

class JigsawPuzzleSA:
    def __init__(self, N, edge_dissimilarity, initial_temp=1000, stopping_temp=1e-3, cooling_rate=0.995, max_iter=10000):
        """
        N: puzzle dimension (NxN puzzle)
        edge_dissimilarity: a dict mapping ((piece1, piece2), direction) -> mismatch value
        initial_temp: starting temperature
        stopping_temp: minimum temperature to stop
        cooling_rate: rate at which temperature decreases
        max_iter: maximum number of iterations
        """
        self.N = N
        self.pieces = list(range(N * N))
        self.edge_dissimilarity = edge_dissimilarity
        self.temp = initial_temp
        self.stopping_temp = stopping_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        
        # best solution found
        self.best_state = None
        self.best_cost = float('inf')
        self.cost_history = []

    def cost(self, state):
        """Compute total edge dissimilarity for a puzzle state"""
        total = 0
        for i in range(self.N):
            for j in range(self.N):
                idx = i*self.N + j
                piece = state[idx]
                # check right neighbor
                if j < self.N - 1:
                    right = state[idx + 1]
                    total += self.edge_dissimilarity.get((piece, right, 'right'), 0)
                # check bottom neighbor
                if i < self.N - 1:
                    bottom = state[idx + self.N]
                    total += self.edge_dissimilarity.get((piece, bottom, 'bottom'), 0)
        return total

    def random_swap(self, state):
        """Swap two random pieces"""
        a, b = random.sample(range(len(state)), 2)
        state[a], state[b] = state[b], state[a]
        return state

    def simulated_annealing(self):
        # start with random configuration
        current_state = self.pieces[:]
        random.shuffle(current_state)
        current_cost = self.cost(current_state)

        self.best_state = current_state[:]
        self.best_cost = current_cost
        self.cost_history.append(current_cost)

        iter_count = 0
        while self.temp > self.stopping_temp and iter_count < self.max_iter:
            candidate = current_state[:]
            candidate = self.random_swap(candidate)
            candidate_cost = self.cost(candidate)

            delta = candidate_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / self.temp):
                current_state = candidate
                current_cost = candidate_cost

                if current_cost < self.best_cost:
                    self.best_state = current_state[:]
                    self.best_cost = current_cost

            self.temp *= self.cooling_rate
            self.cost_history.append(self.best_cost)
            iter_count += 1

        print(f"Best cost: {self.best_cost}")
        print("Best configuration:")
        for i in range(self.N):
            print(self.best_state[i*self.N:(i+1)*self.N])

    def plot_cost(self):
        plt.plot(self.cost_history)
        plt.title("Cost over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Total edge mismatch")
        plt.show()


# ----------------------
# Example usage
# ----------------------

N = 3  # 3x3 puzzle
# Define a dummy edge dissimilarity for simplicity
# Normally, this would be calculated based on piece shapes or colors
edge_dissimilarity = {}
for a in range(N*N):
    for b in range(N*N):
        edge_dissimilarity[(a, b, 'right')] = random.randint(0, 10)
        edge_dissimilarity[(a, b, 'bottom')] = random.randint(0, 10)

puzzle_solver = JigsawPuzzleSA(N, edge_dissimilarity)
puzzle_solver.simulated_annealing()
puzzle_solver.plot_cost()