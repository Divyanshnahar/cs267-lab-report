import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Train the Hopfield network with given patterns."""
        for pattern in patterns:
            bipolar_pattern = self._to_bipolar(pattern)
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        np.fill_diagonal(self.weights, 0)  # Remove self-connections

    def recall(self, pattern, iterations=5):
        """Recall a pattern from the network."""
        state = self._to_bipolar(pattern)
        for _ in range(iterations):
            for i in range(self.size):
                net_input = np.dot(self.weights[i], state)
                state[i] = 1 if net_input > 0 else -1
        return self._to_binary(state)

    def test_capacity(self, patterns):
        """Evaluate the network's ability to recall patterns."""
        self.train(patterns)
        successful_recalls = sum(
            np.array_equal(self.recall(pattern), pattern) for pattern in patterns
        )
        return successful_recalls / len(patterns)

    def _to_bipolar(self, pattern):
        """Convert binary pattern to bipolar."""
        return 2 * pattern - 1

    def _to_binary(self, pattern):
        """Convert bipolar pattern back to binary."""
        return (pattern + 1) // 2

# Parameters
network_size = 100  # Size of the network (10x10)
theoretical_capacity = int(0.15 * network_size)  # Theoretical capacity

# Generate random binary patterns
patterns = [np.random.randint(0, 2, network_size) for _ in range(theoretical_capacity)]

# Test the capacity
hopfield_net = HopfieldNetwork(network_size)
recall_rate = hopfield_net.test_capacity(patterns)

print(f"Theoretical Capacity (P_max): {theoretical_capacity} patterns")
print(f"Recall Rate with {theoretical_capacity} patterns: {recall_rate * 100:.2f}%")