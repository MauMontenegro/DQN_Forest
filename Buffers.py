from collections import deque
import numpy as np
import random as rnd


class Simple_Buffer(object):
    def __init__(self, config):
        self.buffer = deque(maxlen=config.replay_memory_capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*rnd.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer.clear()


# Powers of Prioritized Experience Replay
class Prioritized_Buffer(object):
    def __init__(self, config):
        self.prob_alpha = config.buffer_alpha
        self.capacity = config.replay_memory_capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldest since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.prob_alpha
        P = probs / (probs.sum())

        # gets the indices depending on the probability p
        indices = np.random.choice(len(self.buffer), batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        # Compute importance-sampling weight
        weights = (total * P[indices]) ** (-beta)
        # Normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in list(zip(batch_indices, batch_priorities)):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

    def reset_buffer(self):
        print('Resetting Buffer:')
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
