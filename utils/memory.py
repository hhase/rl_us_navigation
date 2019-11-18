import numpy as np

class ReplayMemory():
    #TODO - change self.memory to np.array | problems with np.array handling while storing non-integer values
    def __init__(self, capacity=0, prioritized_replay=False):
        self.capacity = capacity
        self.memory = list()
        self.push_count = 0
        self.prioritized_replay = prioritized_replay
        self.priorities = np.array([])
        self.rank_priorities = np.array([1/i for i in range(1, self.capacity + 1, 1)])
        self.max_priority = 0.001
        self.probabilities = np.array([])
        self.is_weights = np.array([])
        self.beta = 0.2

    def push(self, experience):
        self.memory.insert(0, experience)

        if len(self.memory) < self.capacity:
            if self.prioritized_replay:
                self.priorities = np.insert(self.priorities, 0, self.max_priority)
                rank_priorities = self.rank_priorities[0:len(self.memory)]
                self.probabilities = rank_priorities/np.sum(rank_priorities)
        else:
            self.memory.pop()

        self.push_count += 1
        if self.prioritized_replay:
            self.update_is_weights()

    def update_priority(self, sample_idx, new_priority):
        self.priorities[sample_idx] = np.abs(new_priority)

    def update_is_weights(self):
        self.is_weights = np.power(len(self.memory) * self.probabilities, -self.beta)
        normalization_weight = np.max(self.is_weights)
        self.is_weights /= normalization_weight

    def get_is_weight(self, sample_idx):
        return self.is_weights[sample_idx]

    def sort_memory(self):
        sorted_idxs = np.flip(np.argsort(self.priorities))
        self.memory = np.array(self.memory)
        self.memory = self.memory[sorted_idxs]
        self.memory = self.memory.tolist()
        self.priorities = self.priorities[sorted_idxs]

    def sample(self, batch_size):
        if self.prioritized_replay:
            idxs = np.random.choice(range(len(self.memory)), batch_size, replace=False, p=self.probabilities)
            batch = [self.memory[idx] for idx in idxs]
            return batch, idxs
        else:
            idxs = np.random.choice(range(len(self.memory)), batch_size, replace=False)
            batch = [self.memory[idx] for idx in idxs]
            return batch, idxs

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
