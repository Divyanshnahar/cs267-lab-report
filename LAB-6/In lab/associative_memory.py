import numpy as np
import matplotlib.pyplot as plt


def to_bipolar(x):
    return 2 * x - 1

def to_binary(x):
    return ((x + 1) // 2).astype(int)


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        self.weights.fill(0)
        for p in patterns:
            bp = to_bipolar(p)
            self.weights += np.outer(bp, bp)
        self.weights /= self.size
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, iterations=10):
        state = to_bipolar(pattern)
        for _ in range(iterations):
            for i in range(self.size):
                h = np.dot(self.weights[i], state)
                state[i] = 1 if h >= 0 else -1
        return to_binary(state)

    def energy(self, state):
        s = to_bipolar(state)
        return -0.5 * s @ (self.weights @ s)



size = 100
hop = HopfieldNetwork(size)

# Random patterns
patterns = [np.random.randint(0, 2, size) for _ in range(3)]
hop.train(patterns)

# Select first pattern and corrupt it (10% noise)
original = patterns[0].copy()
noisy = original.copy()
noise_idx = np.random.choice(size, size // 10, replace=False)
noisy[noise_idx] = 1 - noisy[noise_idx]

recalled = hop.recall(noisy, iterations=20)

# Display
plt.figure(figsize=(10,4))
titles = ["Original", "Noisy", "Recalled"]
images = [original, noisy, recalled]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i].reshape(10,10), cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.show()

print("Energy (original):", hop.energy(original))
print("Energy (noisy):", hop.energy(noisy))
print("Energy (recalled):", hop.energy(recalled))
