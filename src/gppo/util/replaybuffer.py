import random
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import torch
import numpy as np


def _action_tensor(action, device):
    t = torch.tensor(action, device=device) if not isinstance(action, torch.Tensor) else action.to(device)
    return t.float() if t.is_floating_point() else t.long()

def _to_float(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32, device=device)
    return torch.tensor(x, dtype=torch.float32, device=device)


class ReplayBufferBase(ABC):
    @abstractmethod
    def push(self, transition: tuple) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> tuple:
        pass


class ReplayBuffer(ReplayBufferBase):
    """
    Simple replay buffer. Supports both single and parallel environments.
    For n_envs > 1, push accepts batched transitions with leading n_envs dimension.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def _push_single(self, state, action, reward, next_state, done):
        self.buffer.append((
            _to_float(state, self.device),
            _action_tensor(action, self.device),
            _to_float(reward, self.device),
            _to_float(next_state, self.device),
            _to_float(done, self.device),
        ))

    def push(self, transition: tuple) -> None:
        """
        Add transition(s) to the buffer.
        Accepts either:
          - single:  (state, action, reward, next_state, done)  all scalar/1D
          - batched: same but with leading n_envs dimension
        """
        state, action, reward, next_state, done = transition
        reward = np.asarray(reward)
        if reward.ndim >= 1 and reward.shape[0] > 1:
            # batched: iterate over envs
            for i in range(reward.shape[0]):
                self._push_single(state[i], action[i], reward[i], next_state[i], done[i])
        else:
            self._push_single(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones),
        )

    def __len__(self):
        return len(self.buffer)


class ReservoirReplayBuffer(ReplayBuffer):
    """
    Reservoir sampling buffer for continual RL.
    Keeps a uniform sample over all seen transitions.
    Supports parallel environments.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0
        self.device = device

    def _push_single(self, state, action, reward, next_state, done):
        self.n_seen += 1
        traj = (
            _to_float(state, self.device),
            _action_tensor(action, self.device),
            _to_float(reward, self.device),
            _to_float(next_state, self.device),
            _to_float(done, self.device),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(traj)
        else:
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = traj


class TaskBalancedReplayBuffer(ReplayBufferBase):
    """
    Task-balanced replay buffer for multi-task RL.
    Supports parallel environments — expects task_id to be per-env (array or scalar).
    """

    def __init__(self, capacity_per_task: int, device: torch.device):
        self.capacity_per_task = capacity_per_task
        self.buffers = defaultdict(lambda: deque(maxlen=self.capacity_per_task))
        self.device = device

    def _push_single(self, state, action, reward, next_state, done, task_id):
        self.buffers[task_id].append((
            _to_float(state, self.device),
            _action_tensor(action, self.device),
            _to_float(reward, self.device),
            _to_float(next_state, self.device),
            _to_float(done, self.device),
        ))

    def push(self, transition: tuple) -> None:
        """
        Accepts either:
          - single:  (state, action, reward, next_state, done, task_id)
          - batched: same with leading n_envs dimension; task_id is array of ints
        """
        state, action, reward, next_state, done, task_id = transition
        task_id = np.asarray(task_id)
        if task_id.ndim >= 1 and task_id.shape[0] > 1:
            for i in range(task_id.shape[0]):
                self._push_single(state[i], action[i], reward[i], next_state[i], done[i], int(task_id[i]))
        else:
            self._push_single(state, action, reward, next_state, done, int(task_id))

    def sample(self, batch_size: int) -> tuple:
        if not self.buffers:
            raise ValueError("Buffer is empty")
        batch = []
        tasks = list(self.buffers.keys())
        while len(batch) < batch_size:
            task_id = random.choice(tasks)
            if self.buffers[task_id]:
                batch.append(random.choice(list(self.buffers[task_id])))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones),
        )

    def __len__(self):
        return sum(len(buf) for buf in self.buffers.values())
