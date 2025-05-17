import numpy as np
import torch as T

from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    def __init__(self, size, *init_state_args, **init_state_kwargs):
        self.size = size
        self.cntr = 0

        self.action_mem = np.zeros(self.size, dtype=int)
        self.reward_mem = np.zeros(self.size, dtype=np.float32)
        self.done_mem = np.zeros(self.size, dtype=int)

        self._init_state_mem(*init_state_args, **init_state_kwargs)


    @abstractmethod
    def _init_state_mem(self, *args, **kwargs):
        """Initialize `state` and `new_state` memory as arrays"""
        pass

    @abstractmethod
    def _store_state(self, i, state, new_state):
        """Store `state` and `new_state` in their respective memory arrays"""
        pass

    @abstractmethod
    def _sample_state(self, batch_indices, device):
        """Sample `state` and `new_state` given `batch_indices` to index. Move result array to `device`"""
        pass

    def store(self, state, action, reward, new_state, done):
        i = self.cntr % self.size
        
        self._store_state(i, state, new_state)
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.done_mem[i] = done
        self.cntr += 1

    def sample(self, batch_size, device):
        max_size = min(self.cntr, self.size)
        batch_indices = np.random.choice(max_size, batch_size)

        state, new_state = self._sample_state(batch_indices, device)
        actions = T.from_numpy(self.action_mem[batch_indices]).to(device)
        rewards = T.from_numpy(self.reward_mem[batch_indices]).to(device)
        dones = T.from_numpy(self.done_mem[batch_indices]).to(device)
        
        return state, actions, rewards, new_state, dones


    def percent_full(self):
        if self.cntr < self.size:
            return self.cntr / self.size
        else:
            return 1.0
        

class MultiInputReplayBuffer(ReplayBuffer):
    def _init_state_mem(self, grid_shape, num_scalars):
        self.grid_state_mem = np.zeros((self.size, *grid_shape), dtype=np.float32)
        self.scalar_state_mem = np.zeros((self.size, num_scalars), dtype=np.float32)
        self.grid_new_state_mem = np.zeros((self.size, *grid_shape), dtype=np.float32)
        self.scalar_new_state_mem = np.zeros((self.size, num_scalars), dtype=np.float32)


    def _store_state(self, i, state, new_state):
        grid, scalar = state
        self.grid_state_mem[i] = grid
        self.scalar_state_mem[i] = scalar

        new_grid, new_scalar = new_state
        self.grid_new_state_mem[i] = new_grid
        self.scalar_new_state_mem[i] = new_scalar


    def _sample_state(self, batch_indices, device):
        grid = T.from_numpy(self.grid_state_mem[batch_indices]).to(device)
        scalar = T.from_numpy(self.scalar_state_mem[batch_indices]).to(device)
        state = grid, scalar

        new_grid = T.from_numpy(self.grid_new_state_mem[batch_indices]).to(device)
        new_scalar = T.from_numpy(self.scalar_new_state_mem[batch_indices]).to(device)
        new_state = new_grid, new_scalar

        return state, new_state


class DiscreteReplayBuffer(ReplayBuffer):
    def _init_state_mem(self, input_shape):
        self.state_mem = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.new_state_mem = np.zeros((self.size, *input_shape), dtype=np.float32)


    def _store_state(self, i, state, new_state):
        self.state_mem[i] = state
        self.new_state_mem[i] = new_state
        

    def _sample_state(self, batch_indices, device):
        state = T.from_numpy(self.state_mem[batch_indices]).to(device)
        new_state = T.from_numpy(self.new_state_mem[batch_indices]).to(device)
        return state, new_state